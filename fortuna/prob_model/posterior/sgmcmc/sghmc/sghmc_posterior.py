import logging
from typing import Optional
from itertools import cycle
import pathlib

from fortuna.data.loader import DataLoader
from fortuna.prob_model.fit_config.base import FitConfig
from fortuna.prob_model.joint.base import Joint
from fortuna.prob_model.posterior.map.map_trainer import (
    MAPTrainer,
    JittedMAPTrainer,
    MultiDeviceMAPTrainer,
)
from fortuna.prob_model.posterior.map.map_posterior import MAPPosterior
from fortuna.prob_model.posterior.map.map_approximator import \
    MAPPosteriorApproximator
from fortuna.prob_model.posterior.base import Posterior
from fortuna.prob_model.posterior.posterior_multi_state_repository import \
    PosteriorMultiStateRepository
from fortuna.prob_model.posterior.sgmcmc.sgmcmc_posterior_mixin import \
    SGMCMCPosteriorMixin
from fortuna.prob_model.posterior.sgmcmc.sghmc import SGHMC_NAME
from fortuna.prob_model.posterior.sgmcmc.sghmc.sghmc_approximator import \
    SGHMCPosteriorApproximator
from fortuna.prob_model.posterior.sgmcmc.sghmc.sghmc_integrator import \
    sghmc_integrator
from fortuna.prob_model.posterior.sgmcmc.sghmc.sghmc_state import SGHMCState
from fortuna.typing import Status
from fortuna.utils.device import select_trainer_given_devices

logger = logging.getLogger(__name__)


class SGHMCPosterior(SGMCMCPosteriorMixin, Posterior):
    def __init__(
        self,
        joint: Joint,
        posterior_approximator: SGHMCPosteriorApproximator,
    ):
        """
        Stochastic Gradient Hamiltonian Monte Carlo approximate posterior class.

        Parameters
        ----------
        joint: Joint
            A Joint distribution object.
        posterior_approximator: SGHMCPosteriorApproximator
            A SGHMC posterior approximator.
        """
        super().__init__(
            joint=joint, posterior_approximator=posterior_approximator
        )

    def __str__(self):
        return SGHMC_NAME

    def fit(
        self,
        train_data_loader: DataLoader,
        val_data_loader: Optional[DataLoader] = None,
        fit_config: FitConfig = FitConfig(),
        map_fit_config: Optional[FitConfig] = None,
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
            logging.info(f"`FitOptimizer` method in SGHMC is ignored.")

        fit_config.optimizer.method = sghmc_integrator(
            momentum_decay=self.posterior_approximator.momentum_decay,
            momentum_resample_steps=None,
            rng_key=self.rng.get(),
            step_schedule=self.posterior_approximator.step_schedule,
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

        status = {}

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
            map_posterior = MAPPosterior(
                self.joint, posterior_approximator=MAPPosteriorApproximator()
            )
            map_posterior.rng = self.rng
            logging.info("Do a preliminary run of MAP.")
            status["map"] = map_posterior.fit(
                train_data_loader=train_data_loader,
                val_data_loader=val_data_loader,
                fit_config=map_fit_config
                if map_fit_config is not None
                else FitConfig(),
            )
            state = SGHMCState.convert_from_map_state(
                map_state=map_posterior.state.get(),
                optimizer=fit_config.optimizer.method,
            )
            logging.info("Preliminary run with MAP completed.")
        logging.info(f"Run SGHMC.")
        state, status["sghmc"] = trainer.train(
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
        )
        self.state = PosteriorMultiStateRepository(
            size=self.posterior_approximator.n_samples,
            checkpoint_dir=fit_config.checkpointer.save_checkpoint_dir
            if fit_config.checkpointer.dump_state is True
            else None,
        )
        data_loader = cycle(iter(train_data_loader))
        for i in range(self.posterior_approximator.n_samples):
            for _ in range(self.posterior_approximator.n_thinning):
                state, _aux = trainer.training_step(
                    state,
                    next(data_loader),
                    self.joint._batched_log_joint_prob,
                    self.rng.get(),
                    n_train_data,
                )
            if fit_config.checkpointer.save_checkpoint_dir:
                trainer.save_checkpoint(
                    state,
                    pathlib.Path(fit_config.checkpointer.save_checkpoint_dir)
                    / str(i),
                    force_save=True,
                )
            self.state.put(
                state=state,
                i=i,
                keep=fit_config.checkpointer.keep_top_n_checkpoints,
            )
        logging.info("Fit completed.")
        return status
