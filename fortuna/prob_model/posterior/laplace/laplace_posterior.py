from __future__ import annotations

import logging
from typing import Dict, Optional, List, Tuple, Union

import jax.numpy as jnp
from flax.core import FrozenDict
from jax import hessian, lax, vjp, devices, jit, pmap
from jax._src.prng import PRNGKeyArray
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map

from fortuna.data.loader import DataLoader, DeviceDimensionAugmentedLoader
from fortuna.prob_model.fit_config.base import FitConfig
from fortuna.prob_model.joint.base import Joint
from fortuna.prob_model.joint.state import JointState
from fortuna.prob_model.posterior.laplace import LAPLACE_NAME
from fortuna.prob_model.posterior.laplace.laplace_approximator import \
    LaplacePosteriorApproximator
from fortuna.prob_model.posterior.laplace.laplace_state import LaplaceState
from fortuna.prob_model.posterior.run_preliminary_map import run_preliminary_map
from fortuna.prob_model.posterior.map.map_state import MAPState
from fortuna.prob_model.posterior.posterior_state_repository import \
    PosteriorStateRepository
from fortuna.prob_model.prior.gaussian import (DiagonalGaussianPrior,
                                               IsotropicGaussianPrior)
from fortuna.typing import CalibMutable, CalibParams, Mutable, Params, Status, AnyKey
from fortuna.utils.nested_dicts import nested_get, nested_set, nested_unpair
from fortuna.utils.freeze import get_trainable_paths
from fortuna.utils.random import generate_random_normal_like_tree
from fortuna.prob_model.posterior.base import Posterior


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
            An estimate of the posterior standard deviation for each random parameter.
        """
        rav, unravel = ravel_pytree(
            tuple(
                [
                    nested_get(params, keys)
                    for keys in which_params
                ]
            )
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
                lambda v: lax.map(vjp_fn(v[0]), v[1].T), (_batch_inputs[:, None], z)
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

        h = jnp.exp(-self.joint.prior.log_var)
        for i, (batch_inputs, batch_targets) in enumerate(train_data_loader):
            if verbose:
                logging.info(f"Hessian approximation for batch {i + 1}.")
            h += compute_hess_batch(batch_inputs, batch_targets)

        if n_gpu_devices > 0:
            h = jnp.sum(h, 0)

        return unravel(1 / jnp.sqrt(h))

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
                fit_config=fit_config,
                allowed_states=(MAPState, LaplaceState)
            )

        elif super()._should_run_preliminary_map(fit_config, map_fit_config):
            state, status["map"] = run_preliminary_map(
                joint=self.joint,
                train_data_loader=train_data_loader,
                val_data_loader=val_data_loader,
                map_fit_config=map_fit_config,
                rng=self.rng,
                **kwargs
            )
        else:
            raise ValueError("The Laplace approximation must start from a preliminary run of MAP or an existing "
                             "checkpoint or state. Please configure `map_fit_config`, or "
                             "`fit_config.checkpointer.restore_checkpoint_path`, "
                             "or `fit_config.checkpointer.start_from_current_state`.")

        state = self._init_map_state(
            state=state,
            fit_config=fit_config
        )

        if fit_config.optimizer.freeze_fun is not None:
            which_params = get_trainable_paths(
                params=state.params,
                freeze_fun=fit_config.optimizer.freeze_fun
            )
        else:
            which_params = None

        if type(state) == MAPState:
            logging.info("Run the Laplace approximation.")
            std_params = self._gnn_approx(
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
                std=std_params,
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
            if fit_config.checkpointer.dump_state is True
            else None
        )
        self.state.put(state, keep=fit_config.checkpointer.keep_top_n_checkpoints)
        logging.info("Fit completed.")
        return status

    def sample(
        self,
        rng: Optional[PRNGKeyArray] = None,
        **kwargs,
    ) -> JointState:
        if rng is None:
            rng = self.rng.get()
        state = self.state.get()

        if state._which_params is not None:
            mean, std = nested_unpair(
                state.params.unfreeze(),
                state._which_params,
                ("mean", "std"),
            )

            noise = generate_random_normal_like_tree(rng, std)
            params = nested_set(
                d=mean,
                key_paths=state._which_params,
                objs=tuple(
                    [
                        tree_map(
                            lambda m, s, e: m + s * e,
                            nested_get(mean, keys),
                            nested_get(std, keys),
                            nested_get(noise, keys),
                        )
                        for keys in state._which_params
                    ]
                ),
            )
            for k, v in params.items():
                params[k] = FrozenDict(v)
            state = state.replace(params=FrozenDict(params))
        else:
            mean, std = dict(), dict()
            for k, v in state.params.items():
                mean[k] = FrozenDict({"params": v["params"]["mean"]})
                std[k] = FrozenDict({"params": v["params"]["std"]})

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
            self,
            state: Union[MAPState, LaplaceState],
            fit_config: FitConfig
    ) -> MAPState:
        if isinstance(state, LaplaceState):
            if state._which_params is not None:
                state = state.replace(
                    params=FrozenDict(
                        nested_unpair(
                            d=state.params.unfreeze(),
                            key_paths=state._which_params,
                            labels=("mean", "std")
                        )[0]
                    )
                )
            else:
                state = state.replace(
                    params=FrozenDict({k: dict(params=v["params"]["mean"]) for k, v in state.params.items()})
                )

        state = MAPState.init(
            params=state.params,
            mutable=state.mutable,
            optimizer=fit_config.optimizer.method,
            calib_params=state.calib_params,
            calib_mutable=state.calib_mutable,
        )

        return state
