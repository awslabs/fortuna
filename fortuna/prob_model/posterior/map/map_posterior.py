import logging
from typing import Optional

from jax._src.prng import PRNGKeyArray

from fortuna.data.loader import DataLoader
from fortuna.prob_model.fit_config.base import FitConfig
from fortuna.prob_model.joint.base import Joint
from fortuna.prob_model.joint.state import JointState
from fortuna.prob_model.posterior.base import Posterior
from fortuna.prob_model.posterior.map import MAP_NAME
from fortuna.prob_model.posterior.map.map_approximator import \
    MAPPosteriorApproximator
from fortuna.prob_model.posterior.map.map_state import MAPState
from fortuna.prob_model.posterior.map.map_trainer import (
    JittedMAPTrainer, MAPTrainer, MultiDeviceMAPTrainer)
from fortuna.prob_model.posterior.posterior_state_repository import \
    PosteriorStateRepository
from fortuna.typing import Status
from fortuna.utils.device import select_trainer_given_devices

logger = logging.getLogger(__name__)


class MAPPosterior(Posterior):
    def __init__(
        self,
        joint: Joint,
        posterior_approximator: MAPPosteriorApproximator,
    ):
        """
        Maximum-a-Posteriori (MAP) approximate posterior class.

        Parameters
        ----------
        joint: Joint
            A Joint distribution object.
        posterior_approximator: MAPPosteriorApproximator
            A MAP posterior approximator.
        """
        super().__init__(joint=joint, posterior_approximator=posterior_approximator)

    def __str__(self):
        return MAP_NAME

    def fit(
        self,
        train_data_loader: DataLoader,
        val_data_loader: Optional[DataLoader] = None,
        fit_config: FitConfig = FitConfig(),
        **kwargs
    ) -> Status:
        if (
            fit_config.checkpointer.dump_state is True
            and not fit_config.checkpointer.save_checkpoint_dir
        ):
            raise ValueError(
                "`save_checkpoint_dir` must be passed when `dump_state` is set to True."
            )
        (
            init_prob_model_state,
            n_train_data,
            n_val_data,
        ) = self._init(train_data_loader, val_data_loader)

        trainer_cls = select_trainer_given_devices(
            devices=fit_config.processor.devices,
            BaseTrainer=MAPTrainer,
            JittedTrainer=JittedMAPTrainer,
            MultiDeviceTrainer=MultiDeviceMAPTrainer,
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
        )

        if fit_config.checkpointer.restore_checkpoint_path:
            state = self.restore_checkpoint(
                restore_checkpoint_path=fit_config.checkpointer.restore_checkpoint_path,
                optimizer=fit_config.optimizer.method,
            )
        else:
            state = MAPState.init(
                params=init_prob_model_state.params,
                mutable=init_prob_model_state.mutable,
                optimizer=fit_config.optimizer.method,
                calib_params=init_prob_model_state.calib_params,
                calib_mutable=init_prob_model_state.calib_mutable,
            )
        logging.info("Run MAP.")
        state, status = trainer.train(
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
            callbacks=fit_config.callbacks,
        )
        self.state = PosteriorStateRepository(
            fit_config.checkpointer.save_checkpoint_dir
            if fit_config.checkpointer.dump_state is True
            else None
        )
        self.state.put(state, keep=fit_config.checkpointer.keep_top_n_checkpoints)
        logging.info("Fit completed.")
        return status

    def sample(self, rng: Optional[PRNGKeyArray] = None, **kwargs) -> JointState:
        state = self.state.get()
        return JointState(
            params=state.params,
            mutable=state.mutable,
            calib_params=state.calib_params,
            calib_mutable=state.calib_mutable,
        )
