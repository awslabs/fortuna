from __future__ import annotations

import logging
import pathlib
from typing import Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import tqdm
from flax.core import FrozenDict
from flax.training.common_utils import shard, shard_prng_key
from jax import hessian, lax, vjp, devices, jit, pmap
from jax._src.prng import PRNGKeyArray
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map

from fortuna.data.loader import (
    DataLoader,
    DeviceDimensionAugmentedLoader,
)
from fortuna.prob_model.fit_config.base import FitConfig
from fortuna.prob_model.joint.base import Joint
from fortuna.prob_model.joint.state import JointState
from fortuna.prob_model.posterior.base import Posterior
from fortuna.prob_model.posterior.laplace import LAPLACE_NAME
from fortuna.prob_model.posterior.laplace.laplace_approximator import (
    LaplacePosteriorApproximator,
)
from fortuna.prob_model.posterior.laplace.laplace_state import LaplaceState
from fortuna.prob_model.posterior.map.map_state import MAPState
from fortuna.prob_model.posterior.posterior_state_repository import (
    PosteriorStateRepository,
)
from fortuna.prob_model.posterior.run_preliminary_map import run_preliminary_map
from fortuna.prob_model.prior.gaussian import (
    DiagonalGaussianPrior,
    IsotropicGaussianPrior,
)
from fortuna.typing import CalibMutable, CalibParams, Mutable, Params, Status, AnyKey
from fortuna.utils.freeze import get_trainable_paths
from fortuna.utils.nested_dicts import (
    nested_get,
    nested_set,
    nested_unpair,
)
from fortuna.utils.random import generate_random_normal_like_tree
from fortuna.utils.strings import decode_encoded_tuple_of_lists_of_strings_to_array


class LaplacePosterior(Posterior):
    def __init__(
        self,
        joint: Joint,
        posterior_approximator: LaplacePosteriorApproximator,
    ):
        """
        Laplace approximation posterior class.

        Parameters
        ----------
        joint: Joint
            A joint distribution object.
        posterior_approximator: LaplacePosteriorApproximator
            A Laplace posterior approximator.
        """
        super().__init__(joint=joint, posterior_approximator=posterior_approximator)
        if type(joint.prior) not in [DiagonalGaussianPrior, IsotropicGaussianPrior]:
            raise ValueError(
                """The Laplace posterior_approximation is not supported for this model. The prior distribution must be one of the
                following choices: {}.""".format(
                    [DiagonalGaussianPrior, IsotropicGaussianPrior]
                )
            )

    def __str__(self):
        return LAPLACE_NAME

    def _gnn_approx(
        self,
        params: Params,
        train_data_loader: DataLoader,
        mutable: Optional[Mutable] = None,
        calib_params: Optional[CalibParams] = None,
        calib_mutable: Optional[CalibMutable] = None,
        which_params: Optional[Tuple[List[AnyKey, ...]]] = None,
        factorization: str = "diagonal",
        verbose: bool = True,
    ) -> Params:
        """
        Estimate a standard deviation for each parameter using a diagonal Generalized Gauss-Newton Hessian
        approximation.

        Parameters
        ----------
        params : Params
            The random parameters of the probabilistic model.
        train_data_loader: DataLoader
            A training data loader.
        mutable: Optional[Mutable]
            Mutable objects.
        calib_params : Optional[CalibParams]
            The calibration parameters of the probabilistic model.
        calib_mutable : Optional[CalibMutable] = None
            The calibration mutable objects used to evaluate the calibrators.
        which_params : Optional[Tuple[List[AnyKey, ...]]]
            Sequences of keys indicating which parameters to compute the Hessian upon.
        factorization: str = "diagonal"
            Factorization of the GGN approximation. Currently, only "diagonal" is supported.
        verbose: bool
            Whether to log the training progress.

        Returns
        -------
        Params
            An estimate of the likelihood standard deviation for each random parameter.
        """
        rav, unravel = ravel_pytree(
            tuple([nested_get(params, keys) for keys in which_params])
            if which_params
            else params
        )

        def get_params_from_rav(_rav):
            unrav = unravel(_rav)
            return (
                FrozenDict(
                    nested_set(
                        params.unfreeze(),
                        which_params,
                        unrav,
                    )
                )
                if which_params
                else unrav
            )

        def apply_calib_model_manager(_params, _batch_inputs):
            outputs = self.joint.likelihood.model_manager.apply(
                _params, _batch_inputs, mutable=mutable, train=False
            )
            outputs = self.joint.likelihood.output_calib_manager.apply(
                params=calib_params["output_calibrator"]
                if calib_params is not None
                else None,
                mutable=calib_mutable["output_calibrator"]
                if calib_mutable is not None
                else None,
                outputs=outputs,
            )
            return outputs

        apply_fn = lambda _rav, x: apply_calib_model_manager(
            get_params_from_rav(_rav), x
        ).squeeze(0)
        vjp_fn = lambda x: vjp(lambda _rav: apply_fn(_rav, x), rav)[1]

        def eig_hess_fn(vars):
            hess = hessian(
                lambda __o: self.joint.likelihood.prob_output_layer.log_prob(
                    __o, vars[1]
                )
            )(vars[0])
            return jnp.linalg.eigh(hess)

        def compute_hess_batch(_batch_inputs, _batch_targets):
            lam, z = lax.map(
                eig_hess_fn,
                (apply_calib_model_manager(params, _batch_inputs), _batch_targets),
            )
            ztj = lax.map(
                lambda v: lax.map(vjp_fn(v[0]), v[1].T),
                (tree_map(lambda x: x[:, None], _batch_inputs), z),
            )[0]
            if factorization == "diagonal":
                return -jnp.sum(lam[:, :, None] * ztj**2, (0, 1))
            raise ValueError(
                f"`factorization={factorization}` not recognized. Currently, only "
                f"`factorization='diagonal'` is supported."
            )

        n_gpu_devices = len([d for d in devices() if d.platform == "gpu"])
        if n_gpu_devices > 0:
            train_data_loader = DeviceDimensionAugmentedLoader(train_data_loader)
            compute_hess_batch = pmap(compute_hess_batch, axis_name="batch")
        else:
            compute_hess_batch = jit(compute_hess_batch)

        h = 0.0
        for i, (batch_inputs, batch_targets) in enumerate(train_data_loader):
            if verbose:
                logging.info(f"Hessian approximation for batch {i + 1}.")
            h += compute_hess_batch(batch_inputs, batch_targets)

        if n_gpu_devices > 0:
            h = jnp.sum(h, 0)
        return unravel(h)

    def _compute_std(self, prior_log_var: float, hess_lik_diag: Params) -> Params:
        hess_prior = jnp.exp(-prior_log_var)
        hess_lik_diag_rav, unravel = ravel_pytree(hess_lik_diag)
        return unravel(1 / jnp.sqrt(hess_prior + hess_lik_diag_rav))

    def fit(
        self,
        train_data_loader: DataLoader,
        val_data_loader: Optional[DataLoader] = None,
        fit_config: FitConfig = FitConfig(),
        map_fit_config: Optional[FitConfig] = None,
        **kwargs,
    ) -> Dict[str, Status]:
        super()._checks_on_fit_start(fit_config, map_fit_config)

        status = dict()

        if super()._is_state_available_somewhere(fit_config):
            state = super()._restore_state_from_somewhere(
                fit_config=fit_config, allowed_states=(MAPState, LaplaceState)
            )

        elif super()._should_run_preliminary_map(fit_config, map_fit_config):
            state, status["map"] = run_preliminary_map(
                joint=self.joint,
                train_data_loader=train_data_loader,
                val_data_loader=val_data_loader,
                map_fit_config=map_fit_config,
                rng=self.rng,
                **kwargs,
            )
        else:
            raise ValueError(
                "The Laplace approximation must start from a preliminary run of MAP or an existing "
                "checkpoint or state. Please configure `map_fit_config`, or "
                "`fit_config.checkpointer.restore_checkpoint_path`, "
                "or `fit_config.checkpointer.start_from_current_state`."
            )

        state = self._init_map_state(state=state, fit_config=fit_config)

        if fit_config.optimizer.freeze_fun is not None:
            which_params = get_trainable_paths(
                params=state.params, freeze_fun=fit_config.optimizer.freeze_fun
            )
        else:
            which_params = None

        logging.info("Run the Laplace approximation.")
        hess_lik_diag = self._gnn_approx(
            state.params,
            train_data_loader,
            mutable=state.mutable,
            calib_params=state.calib_params,
            calib_mutable=state.calib_mutable,
            which_params=which_params,
            verbose=fit_config.monitor.verbose,
        )
        state = LaplaceState.convert_from_map_state(
            map_state=state,
            hess_lik_diag=hess_lik_diag,
            prior_log_var=self.joint.prior.log_var,
            which_params=which_params,
        )

        if fit_config.checkpointer.save_checkpoint_dir:
            self.save_checkpoint(
                state,
                save_checkpoint_dir=fit_config.checkpointer.save_checkpoint_dir,
                keep=fit_config.checkpointer.keep_top_n_checkpoints,
                force_save=True,
            )

        self.state = PosteriorStateRepository(
            fit_config.checkpointer.save_checkpoint_dir
        )
        self.state.put(state, keep=fit_config.checkpointer.keep_top_n_checkpoints)
        logging.info("Fit completed.")
        if val_data_loader is not None and self.posterior_approximator.tune_prior_log_variance:
            logging.info("Tuning the prior log-variance now")
            opt_prior_log_var = self.prior_log_variance_tuning(
                val_data_loader=val_data_loader,
                n_posterior_samples=5,
                distribute=fit_config.processor.devices == -1,
            )
            state = state.replace(prior_log_var=opt_prior_log_var)
            self.state.put(state, keep=fit_config.checkpointer.keep_top_n_checkpoints)
            logging.info(f"Best prior log-variance found: {opt_prior_log_var}")
        return status

    def sample(
        self,
        rng: Optional[PRNGKeyArray] = None,
        **kwargs,
    ) -> JointState:
        if rng is None:
            rng = self.rng.get()
        state: LaplaceState = self.state.get()
        if kwargs.get("prior_log_var") is not None:
            state = state.replace(prior_log_var=kwargs.get("prior_log_var"))

        if state._encoded_which_params is not None:
            which_params = decode_encoded_tuple_of_lists_of_strings_to_array(
                state._encoded_which_params
            )
            mean, hess_lik_diag = nested_unpair(
                state.params.unfreeze(),
                which_params,
                ("mean", "hess_lik_diag"),
            )
            std = self._compute_std(
                prior_log_var=state.prior_log_var, hess_lik_diag=hess_lik_diag
            )

            noise = generate_random_normal_like_tree(rng, std)
            params = nested_set(
                d=mean,
                key_paths=which_params,
                objs=tuple(
                    [
                        tree_map(
                            lambda m, s, e: m + s * e,
                            nested_get(mean, keys),
                            nested_get(std, keys),
                            nested_get(noise, keys),
                        )
                        for keys in which_params
                    ]
                ),
            )
            for k, v in params.items():
                params[k] = FrozenDict(v)
            state = state.replace(params=FrozenDict(params))
        else:
            mean, hess_lik_diag = dict(), dict()
            for k, v in state.params.items():
                mean[k] = FrozenDict({"params": v["params"]["mean"]})
                hess_lik_diag[k] = FrozenDict({"params": v["params"]["hess_lik_diag"]})

            std = self._compute_std(
                prior_log_var=state.prior_log_var, hess_lik_diag=hess_lik_diag
            )
            state = state.replace(
                params=FrozenDict(
                    tree_map(
                        lambda m, s, e: m + s * e,
                        mean,
                        std,
                        generate_random_normal_like_tree(rng, std),
                    )
                )
            )

        return JointState(
            params=state.params,
            mutable=state.mutable,
            calib_params=state.calib_params,
            calib_mutable=state.calib_mutable,
        )

    def _init_map_state(
        self, state: Union[MAPState, LaplaceState], fit_config: FitConfig
    ) -> MAPState:
        if isinstance(state, LaplaceState):
            if state._encoded_which_params is not None:
                which_params = decode_encoded_tuple_of_lists_of_strings_to_array(
                    state._encoded_which_params
                )
                state = state.replace(
                    params=FrozenDict(
                        nested_unpair(
                            d=state.params.unfreeze(),
                            key_paths=which_params,
                            labels=("mean", "hess_lik_diag"),
                        )[0]
                    )
                )
            else:
                state = state.replace(
                    params=FrozenDict(
                        {
                            k: dict(params=v["params"]["mean"])
                            for k, v in state.params.items()
                        }
                    )
                )

        state = MAPState.init(
            params=state.params,
            mutable=state.mutable,
            optimizer=fit_config.optimizer.method,
            calib_params=state.calib_params,
            calib_mutable=state.calib_mutable,
        )

        return state

    def _batched_log_prob(
        self,
        batch,
        prior_log_var: float,
        n_posterior_samples: int = 30,
        rng: Optional[PRNGKeyArray] = None,
        **kwargs,
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, dict]]:
        import jax.random as random
        import jax.scipy as jsp

        if rng is None:
            rng = self.rng.get()
        keys = random.split(rng, n_posterior_samples)

        def _lik_log_batched_prob(key):
            sample = self.sample(inputs=batch[0], rng=key, prior_log_var=prior_log_var)
            return self.joint.likelihood._batched_log_prob(
                sample.params,
                batch,
                mutable=sample.mutable,
                calib_params=sample.calib_params,
                calib_mutable=sample.calib_mutable,
                **kwargs,
            )

        return jsp.special.logsumexp(
            lax.map(_lik_log_batched_prob, keys), axis=0
        ) - jnp.log(n_posterior_samples)

    def prior_log_variance_tuning(
        self,
        val_data_loader: DataLoader,
        n_posterior_samples: int = 10,
        mode: str = "cv",
        min_prior_log_var: float = -3,
        max_prior_log_var: float = 3,
        grid_size: int = 20,
        distribute: bool = False,
    ) -> jnp.ndarray:
        if mode == "cv":
            return self._prior_log_variance_tuning_cv(
                val_data_loader,
                n_posterior_samples,
                min_prior_log_var,
                max_prior_log_var,
                grid_size,
                distribute,
            )
        elif mode == "marginal_lik":
            raise NotImplementedError(
                f"Optimizing the prior log variance via marginal likelihood maximization is not yet available."
            )
        else:
            raise ValueError(f"Unrecognized mode={mode} for prior log variance tuning.")

    def _prior_log_variance_tuning_cv(
        self,
        val_data_loader: DataLoader,
        n_posterior_samples: int,
        min_prior_log_var: float,
        max_prior_log_var: float,
        grid_size: int,
        distribute: bool,
    ) -> jnp.ndarray:
        best = None
        candidates = list(
            jnp.linspace(min_prior_log_var, max_prior_log_var, grid_size)
        ) + [jnp.array(self.joint.prior.log_var)]
        if distribute:
            rng = shard_prng_key(jax.random.PRNGKey(0))
            val_data_loader = DeviceDimensionAugmentedLoader(val_data_loader)
            candidates = [shard(c) for c in candidates]
            fn = pmap(self._batched_log_prob, static_broadcasted_argnums=(2,))
        else:
            fn = jit(self._batched_log_prob, static_argnums=(2,))

        for lpv in tqdm.tqdm(candidates, desc="Tuning prior log-var"):
            neg_log_prob = -jnp.sum(
                jnp.concatenate(
                    [
                        self.joint.likelihood._unshard_array(
                            fn(batch, lpv, n_posterior_samples, rng)
                        )
                        for batch in val_data_loader
                    ],
                    0,
                )
            )
            if best is None or neg_log_prob < best[-1]:
                best = (lpv, neg_log_prob)

        opt_prior_log_var = best[0].reshape()
        return opt_prior_log_var
