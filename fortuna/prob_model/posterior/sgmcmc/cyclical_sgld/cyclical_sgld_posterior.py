import logging
from typing import Optional
import pathlib

from fortuna.data.loader import DataLoader
from fortuna.prob_model.fit_config.base import FitConfig
from fortuna.prob_model.joint.base import Joint
from fortuna.prob_model.posterior.map.map_trainer import (
    MAPTrainer,
    JittedMAPTrainer,
    MultiDeviceMAPTrainer,
)
from fortuna.prob_model.posterior.base import Posterior
from fortuna.prob_model.posterior.posterior_multi_state_repository import \
    PosteriorMultiStateRepository
from fortuna.prob_model.posterior.sgmcmc.sgmcmc_posterior_mixin import \
    SGMCMCPosteriorMixin
from fortuna.prob_model.posterior.sgmcmc.cyclical_sgld import \
    CYCLICAL_SGLD_NAME
from fortuna.prob_model.posterior.sgmcmc.cyclical_sgld.cyclical_sgld_approximator import \
    CyclicalSGLDPosteriorApproximator
from fortuna.prob_model.posterior.sgmcmc.cyclical_sgld.cyclical_sgld_integrator import \
    cyclical_sgld_integrator
from fortuna.prob_model.posterior.sgmcmc.cyclical_sgld.cyclical_sgld_callback import \
    CyclicalSGLDSamplingCallback
from fortuna.prob_model.posterior.sgmcmc.cyclical_sgld.cyclical_sgld_state import \
    CyclicalSGLDState
from fortuna.typing import Status
from fortuna.utils.device import select_trainer_given_devices

logger = logging.getLogger(__name__)


class CyclicalSGLDPosterior(SGMCMCPosteriorMixin, Posterior):
    def __init__(
        self,
        joint: Joint,
        posterior_approximator: CyclicalSGLDPosteriorApproximator,
    ):
        """
        Cyclical Stochastic Gradient Langevin Dynamics (SGLD) approximate posterior class.

        Parameters
        ----------
        joint: Joint
            A Joint distribution object.
        posterior_approximator: CyclicalSGLDPosteriorApproximator
            A cyclical SGLD posterior approximator.
        """
        super().__init__(
            joint=joint, posterior_approximator=posterior_approximator
        )

    def __str__(self):
        return CYCLICAL_SGLD_NAME

    def fit(
        self,
        train_data_loader: DataLoader,
        val_data_loader: Optional[DataLoader] = None,
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
        (
            init_prob_model_state,
            n_train_data,
            n_val_data,
        ) = self._init(train_data_loader, val_data_loader)

        if fit_config.optimizer.method is not None:
            logging.info(f"`FitOptimizer` method in CyclicalSGLD is ignored.")

        fit_config.optimizer.method = cyclical_sgld_integrator(
            rng_key=self.rng.get(),
            init_step_size=self.posterior_approximator.init_step_size,
            cycle_length=self.posterior_approximator.cycle_length,
            exploration_ratio=self.posterior_approximator.exploration_ratio,
            preconditioner=self.posterior_approximator.preconditioner,
        )

        trainer_cls = select_trainer_given_devices(
            devices=fit_config.processor.devices,
            BaseTrainer=MAPTrainer,
            JittedTrainer=JittedMAPTrainer,
            MultiDeviceTrainer=MultiDeviceMAPTrainer,
            disable_jit=fit_config.processor.disable_jit,
        )

        save_checkpoint_dir = (
            pathlib.Path(fit_config.checkpointer.save_checkpoint_dir) / "c"
            if fit_config.checkpointer.save_checkpoint_dir
            else None
        )
        trainer = trainer_cls(
            predict_fn=self.joint.likelihood.prob_output_layer.predict,
            save_checkpoint_dir=save_checkpoint_dir,
            save_every_n_steps=fit_config.checkpointer.save_every_n_steps,
            keep_top_n_checkpoints=fit_config.checkpointer.keep_top_n_checkpoints,
            disable_training_metrics_computation=fit_config.monitor.disable_training_metrics_computation,
            eval_every_n_epochs=fit_config.monitor.eval_every_n_epochs,
            early_stopping_monitor=fit_config.monitor.early_stopping_monitor,
            early_stopping_min_delta=fit_config.monitor.early_stopping_min_delta,
            early_stopping_patience=fit_config.monitor.early_stopping_patience,
        )

        if fit_config.checkpointer.restore_checkpoint_path:
            restore_checkpoint_path = (
                pathlib.Path(fit_config.checkpointer.restore_checkpoint_path)
                / "c"
            )
            state = self.restore_checkpoint(
                restore_checkpoint_path=restore_checkpoint_path,
                optimizer=fit_config.optimizer.method,
            )
        else:
            state = CyclicalSGLDState.init(
                params=init_prob_model_state.params,
                mutable=init_prob_model_state.mutable,
                optimizer=fit_config.optimizer.method,
                calib_params=init_prob_model_state.calib_params,
                calib_mutable=init_prob_model_state.calib_mutable,
            )
        self.state = PosteriorMultiStateRepository(
            size=self.posterior_approximator.n_samples,
            checkpoint_dir=fit_config.checkpointer.save_checkpoint_dir
            if fit_config.checkpointer.dump_state is True
            else None,
        )

        cyclical_sampling_callback = CyclicalSGLDSamplingCallback(
            n_epochs=fit_config.optimizer.n_epochs,
            n_samples=self.posterior_approximator.n_samples,
            n_thinning=self.posterior_approximator.n_thinning,
            cycle_length=self.posterior_approximator.cycle_length,
            exploration_ratio=self.posterior_approximator.exploration_ratio,
            trainer=trainer,
            state_repository=self.state,
            keep_top_n_checkpoints=fit_config.checkpointer.keep_top_n_checkpoints,
            save_checkpoint_dir=fit_config.checkpointer.save_checkpoint_dir,
        )

        logging.info(f"Run CyclicalSGLD.")
        state, status = trainer.train(
            rng=self.rng.get(),
            state=state,
            loss_fun=self.joint._batched_log_joint_prob,
            training_dataloader=train_data_loader,
            training_dataset_size=n_train_data,
            n_epochs=fit_config.optimizer.n_epochs,
            metrics=fit_config.monitor.metrics,
            validation_dataloader=val_data_loader,
            validation_dataset_size=n_val_data,
            verbose=fit_config.monitor.verbose,
            callbacks=[cyclical_sampling_callback],
        )
        logging.info("Fit completed.")

        if cyclical_sampling_callback.samples_count < self.posterior_approximator.n_samples:
            raise RuntimeError(f"The number of sampled states {cyclical_sampling_callback.samples_count} "
                               "is less than the desired number of samples "
                               f"{self.posterior_approximator.n_samples}. Consider adjusting the cycle "
                               "length or the thinning parameter.")
        return status
