from __future__ import annotations

import logging
from typing import Dict, Optional

import jax.numpy as jnp
from flax.core import FrozenDict
from jax import hessian, lax, vjp, devices, jit, pmap
from jax._src.prng import PRNGKeyArray
from jax.flatten_util import ravel_pytree

from fortuna.data.loader import DataLoader, DeviceDimensionAugmentedDataLoader
from fortuna.prob_model.fit_config.base import FitConfig
from fortuna.prob_model.joint.base import Joint
from fortuna.prob_model.joint.state import JointState
from fortuna.prob_model.posterior.gaussian_posterior import GaussianPosterior
from fortuna.prob_model.posterior.laplace import LAPLACE_NAME
from fortuna.prob_model.posterior.laplace.laplace_approximator import \
    LaplacePosteriorApproximator
from fortuna.prob_model.posterior.laplace.laplace_state import LaplaceState
from fortuna.prob_model.posterior.map.map_approximator import \
    MAPPosteriorApproximator
from fortuna.prob_model.posterior.map.map_posterior import MAPPosterior
from fortuna.prob_model.posterior.map.map_state import MAPState
from fortuna.prob_model.posterior.posterior_state_repository import \
    PosteriorStateRepository
from fortuna.prob_model.prior.gaussian import (DiagonalGaussianPrior,
                                               IsotropicGaussianPrior)
from fortuna.typing import CalibMutable, CalibParams, Mutable, Params, Status
from fortuna.utils.nested_dicts import nested_get, nested_set


class LaplacePosterior(GaussianPosterior):
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
                    for keys in self.posterior_approximator.which_params
                ]
            )
            if self.posterior_approximator.which_params
            else params
        )

        def get_params_from_rav(_rav):
            unrav = unravel(_rav)
            return (
                FrozenDict(
                    nested_set(
                        params.unfreeze(),
                        self.posterior_approximator.which_params,
                        unrav,
                    )
                )
                if self.posterior_approximator.which_params
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
            train_data_loader = DeviceDimensionAugmentedDataLoader(train_data_loader)
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
        if (
            fit_config.checkpointer.dump_state is True
            and not fit_config.checkpointer.save_checkpoint_dir
        ):
            raise ValueError(
                "`save_checkpoint_dir` must be passed when `dump_state` is set to True."
            )

        status = {}

        if not fit_config.checkpointer.restore_checkpoint_path:
            map_posterior = MAPPosterior(
                self.joint, posterior_approximator=MAPPosteriorApproximator()
            )
            map_posterior.rng = self.rng
            logging.info("Do a preliminary run of MAP.")
            status["map"] = map_posterior.fit(
                rng=self.rng.get(),
                train_data_loader=train_data_loader,
                val_data_loader=val_data_loader,
                fit_config=map_fit_config
                if map_fit_config is not None
                else FitConfig(),
                **kwargs,
            )
            logging.info("Preliminary run with MAP completed.")
            state = map_posterior.state.get()
        else:
            state = self.restore_checkpoint(
                restore_checkpoint_path=fit_config.checkpointer.restore_checkpoint_path
            )

        if type(state) == MAPState:
            logging.info("Run the Laplace approximation.")
            std_params = self._gnn_approx(
                state.params,
                train_data_loader,
                mutable=state.mutable,
                calib_params=state.calib_params,
                calib_mutable=state.calib_mutable,
                verbose=fit_config.monitor.verbose,
            )
            state = LaplaceState.convert_from_map_state(
                map_state=state,
                std=std_params,
                which_params=self.posterior_approximator.which_params,
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
        return self._sample_diag_gaussian(rng=rng, **kwargs)
