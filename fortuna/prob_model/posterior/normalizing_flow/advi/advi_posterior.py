from __future__ import annotations

import logging
from typing import Optional, List

import jax.numpy as jnp
from flax.core import FrozenDict
from jax._src.prng import PRNGKeyArray
from jax.flatten_util import ravel_pytree

from fortuna.data.loader import DataLoader
from fortuna.distribution.gaussian import DiagGaussian
from fortuna.prob_model.fit_config.base import FitConfig
from fortuna.prob_model.joint.base import Joint
from fortuna.prob_model.joint.state import JointState
from fortuna.prob_model.posterior.base import Posterior
from fortuna.prob_model.posterior.normalizing_flow.advi import ADVI_NAME
from fortuna.prob_model.posterior.normalizing_flow.advi.advi_approximator import \
    ADVIPosteriorApproximator
from fortuna.prob_model.posterior.normalizing_flow.advi.advi_architecture import \
    ADVIArchitecture
from fortuna.prob_model.posterior.normalizing_flow.advi.advi_state import \
    ADVIState
from fortuna.prob_model.posterior.normalizing_flow.advi.advi_trainer import \
    ADVITrainer
from fortuna.prob_model.posterior.posterior_state_repository import \
    PosteriorStateRepository
from fortuna.training.trainer import JittedMixin, MultiDeviceMixin
from fortuna.typing import Status
from fortuna.utils.device import select_trainer_given_devices
from fortuna.prob_model.posterior.map.map_state import MAPState
from fortuna.utils.nested_dicts import nested_get, nested_unpair
from fortuna.prob_model.posterior.map.map_posterior import MAPPosterior
from fortuna.prob_model.posterior.map.map_approximator import MAPPosteriorApproximator


class JittedADVITrainer(JittedMixin, ADVITrainer):
    pass


class MultiDeviceADVITrainer(MultiDeviceMixin, ADVITrainer):
    pass


class ADVIPosterior(Posterior):
    def __init__(
        self,
        joint: Joint,
        posterior_approximator: ADVIPosteriorApproximator,
    ):
        """
        Automatic Differentiation Variational Inference (ADVI) approximate posterior class.

        Parameters
        ----------
        joint: Joint
            A joint distribution object.
        posterior_approximator: ADVI
            An ADVI posterior approximator.
        """
        super().__init__(joint=joint, posterior_approximator=posterior_approximator)

    def __str__(self):
        return ADVI_NAME

    def fit(
        self,
        train_data_loader: DataLoader,
        val_data_loader: Optional[DataLoader] = None,
        map_fit_config: Optional[FitConfig] = None,
        fit_config: FitConfig = FitConfig(),
        **kwargs,
    ) -> Status:
        if (
            fit_config.checkpointer.dump_state is True
            and not fit_config.checkpointer.save_checkpoint_dir
        ):
            raise ValueError(
                "`save_checkpoint_dir` must be passed when `dump_state` is set to True."
            )

        state = None
        stds = None
        allowed_states = [MAPState, ADVIState]
        status = dict()

        if fit_config.checkpointer.restore_checkpoint_path is not None:
            state = self.restore_checkpoint(
                restore_checkpoint_path=fit_config.checkpointer.restore_checkpoint_path,
                optimizer=fit_config.optimizer.method,
            )

            if type(state) not in allowed_states:
                raise ValueError(f"The type of the restored checkpoint must be within {allowed_states}. "
                                 f"However, {fit_config.checkpointer.restore_checkpoint_path} pointed to a state "
                                 f"with type {type(state)}.")

            if type(state) == ADVIState:
                means, stds = nested_unpair(
                    state.params.unfreeze(),
                    self.posterior_approximator.which_params,
                    ("mean", "std")
                ) if self.posterior_approximator.which_params is not None else \
                    [FrozenDict({k: dict(params=v["params"][s]) for k, v in state.params.items()}) for s in ["mean", "std"]]
                means, stds = FrozenDict(means), FrozenDict(stds)
                state = state.replace(params=means)
                del means
        elif map_fit_config is not None:
            map_posterior = MAPPosterior(
                self.joint, posterior_approximator=MAPPosteriorApproximator()
            )
            map_posterior.rng = self.rng
            logging.info("Do a preliminary run of MAP.")
            status["map"] = map_posterior.fit(
                rng=self.rng.get(),
                train_data_loader=train_data_loader,
                val_data_loader=val_data_loader,
                fit_config=map_fit_config,
                **kwargs,
            )
            logging.info("Preliminary run with MAP completed.")
            state = map_posterior.state.get()

        state, n_train_data, n_val_data = self._init(
            train_data_loader, val_data_loader, state
        )

        if self.posterior_approximator.which_params is None:
            rav, self._unravel = ravel_pytree(state.params)
            rav_stds = ravel_pytree(stds)[0] if stds is not None else None
        else:
            def unravel_fn(_params, _path):
                return ravel_pytree(nested_get(_params, _path))
            rav, self._unravel, sizes, rav_stds = [], [], [], []
            for path in self.posterior_approximator.which_params:
                _rav, _unravel = unravel_fn(state.params, path)
                self._unravel.append(_unravel)
                rav.append(_rav)
                if stds is not None:
                    rav_stds.append(unravel_fn(stds, path)[0])
                sizes.append(len(_rav))
            rav = jnp.concatenate(rav)
            if stds is not None:
                rav_stds = jnp.concatenate(rav_stds)
            self._unravel = tuple(self._unravel)

        size_rav = len(rav)
        self.base = DiagGaussian(
            mean=jnp.zeros(size_rav),
            std=self.posterior_approximator.std_base * jnp.ones(size_rav),
        )
        self.architecture = ADVIArchitecture(
            size_rav, std_init_params=self.posterior_approximator.std_init_params
        )

        trainer_cls = select_trainer_given_devices(
            devices=fit_config.processor.devices,
            BaseTrainer=ADVITrainer,
            JittedTrainer=JittedADVITrainer,
            MultiDeviceTrainer=MultiDeviceADVITrainer,
            disable_jit=fit_config.processor.disable_jit,
        )

        trainer = trainer_cls(
            predict_fn=self.joint.likelihood.prob_output_layer.predict,
            save_checkpoint_dir=fit_config.checkpointer.save_checkpoint_dir,
            save_every_n_steps=fit_config.checkpointer.save_every_n_steps,
            keep_top_n_checkpoints=fit_config.checkpointer.keep_top_n_checkpoints,
            disable_training_metrics_computation=fit_config.monitor.disable_training_metrics_computation,
            eval_every_n_epochs=fit_config.monitor.eval_every_n_epochs,
            early_stopping_monitor=fit_config.monitor.early_stopping_monitor,
            early_stopping_min_delta=fit_config.monitor.early_stopping_min_delta,
            early_stopping_patience=fit_config.monitor.early_stopping_patience,
            base=self.base,
            architecture=self.architecture,
            which_params=self.posterior_approximator.which_params,
            all_params=state.params if self.posterior_approximator.which_params else None,
            sizes=sizes if self.posterior_approximator.which_params else None,
            unravel=self._unravel
        )

        state = ADVIState.init(
            FrozenDict(
                zip(
                    ("mean", "logvar"),
                    trainer.init_params(
                        self.rng.get(),
                        mean=rav
                    ) if stds is None else (rav, 2 * jnp.log(rav_stds)),
                )
            ),
            getattr(state, "mutable", state.mutable),
            fit_config.optimizer.method,
            getattr(state, "calib_params", state.calib_params),
            getattr(state, "calib_mutable", state.calib_mutable),
        )

        logging.info("Run ADVI.")
        state, status["advi"] = trainer.train(
            rng=self.rng.get(),
            state=state,
            loss_fun=self.joint._batched_negative_log_joint_prob,
            training_dataloader=train_data_loader,
            training_dataset_size=n_train_data,
            n_epochs=fit_config.optimizer.n_epochs,
            metrics=fit_config.monitor.metrics,
            validation_dataloader=val_data_loader,
            validation_dataset_size=n_val_data,
            verbose=fit_config.monitor.verbose,
            unravel=self._unravel,
            n_samples=self.posterior_approximator.n_loss_samples,
            callbacks=fit_config.callbacks,
        )
        trainer._all_params = None

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
