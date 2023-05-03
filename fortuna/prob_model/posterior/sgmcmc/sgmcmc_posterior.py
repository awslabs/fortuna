import logging
from typing import Optional
from itertools import cycle
import pathlib

from jax._src.prng import PRNGKeyArray
from flax.core import FrozenDict
from jax import pure_callback, random

from fortuna.typing import Array
from fortuna.data.loader import DataLoader, InputsLoader
from fortuna.prob_model.fit_config.base import FitConfig
from fortuna.prob_model.joint.base import Joint
from fortuna.prob_model.joint.state import JointState
from fortuna.prob_model.posterior.map.map_trainer import (
    MAPTrainer,
    JittedMAPTrainer,
    MultiDeviceMAPTrainer,
)
from fortuna.prob_model.posterior.map.map_state import \
    MAPState
from fortuna.prob_model.posterior.map.map_posterior import \
    MAPPosterior
from fortuna.prob_model.posterior.map.map_approximator import \
    MAPPosteriorApproximator
from fortuna.prob_model.posterior.base import Posterior
from fortuna.prob_model.posterior.base import PosteriorApproximator
from fortuna.prob_model.posterior.posterior_multi_state_repository import \
    PosteriorMultiStateRepository
from fortuna.typing import Path, Status
from fortuna.utils.device import select_trainer_given_devices

logger = logging.getLogger(__name__)


class SGMCMCPosterior(Posterior):
    def __init__(
        self,
        joint: Joint,
        posterior_approximator: PosteriorApproximator,
    ):
        """
        An abstract approximate posterior class for SG-MCMC methods that keeps a repository
        of frozen parameter samples after the initial burn-in phase.

        Parameters
        ----------
        joint: Joint
            A Joint distribution object.
        posterior_approximator: PosteriorApproximator
            A posterior approximator.
        """
        super().__init__(
            joint=joint, posterior_approximator=posterior_approximator
        )

    def __str__(self):
        raise NotImplementedError

    def get_integrator(self):
        raise NotImplementedError

    def convert_state_from_map_state(self, *args, **kwargs):
        raise NotImplementedError

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
        if (
            fit_config.checkpointer.restore_checkpoint_path is not None
            and map_fit_config is not None
        ):
            raise ValueError(
                "Setting `fit_config.checkpointer.restore_checkpoint_path` is incompatible with `map_fit_config`."
            )

        (
            init_prob_model_state,
            n_train_data,
            n_val_data,
        ) = self._init(train_data_loader, val_data_loader)

        if fit_config.optimizer.method is not None:
            logging.info(f"`FitOptimizer` method in {str(self)} is ignored.")

        fit_config.optimizer.method = self.get_integrator()

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
            restore_checkpoint_path = \
                pathlib.Path(fit_config.checkpointer.restore_checkpoint_path) / "c"
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
            state = self.convert_state_from_map_state(
                map_state=map_posterior.state.get(),
                optimizer=fit_config.optimizer.method,
            )
            logging.info("Preliminary run with MAP completed.")
        logging.info(f"Run {str(self)}.")
        state, status[str(self)] = trainer.train(
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
            size=self.posterior_approximator.num_samples,
            checkpoint_dir=fit_config.checkpointer.save_checkpoint_dir
            if fit_config.checkpointer.dump_state is True
            else None
        )
        data_loader = cycle(iter(train_data_loader))
        for i in range(self.posterior_approximator.num_samples):
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
                    pathlib.Path(fit_config.checkpointer.save_checkpoint_dir) / str(i),
                    force_save=True
                )
            self.state.put(
                state=state, i=i, keep=fit_config.checkpointer.keep_top_n_checkpoints
            )
        logging.info("Fit completed.")
        return status

    def sample(
        self,
        rng: Optional[PRNGKeyArray] = None,
        **kwargs,
    ) -> JointState:
        """
        Sample from the posterior distribution.

        Parameters
        ----------
        rng : Optional[PRNGKeyArray]
            A random number generator. If not passed, this will be taken from the attributes of this class.

        Returns
        -------
        JointState
            A sample from the posterior distribution.
        """
        if rng is None:
            rng = self.rng.get()
        state = pure_callback(
            lambda j: self.state.get(i=j),
            self.state.get(i=0),
            random.choice(rng, self.posterior_approximator.num_samples),
        )
        return JointState(
            params=state.params,
            mutable=state.mutable,
            calib_params=state.calib_params,
            calib_mutable=state.calib_mutable,
        )

    def load_state(self, checkpoint_dir: Path) -> None:
        try:
            self.restore_checkpoint(pathlib.Path(checkpoint_dir) / "0")
        except ValueError:
            raise ValueError(
                f"No checkpoint was found in `checkpoint_dir={checkpoint_dir}`."
            )
        self.state = PosteriorMultiStateRepository(
            size=self.posterior_approximator.num_samples,
            checkpoint_dir=checkpoint_dir,
        )

    def save_state(self, checkpoint_dir: Path, keep_top_n_checkpoints: int = 1) -> None:
        for i in range(self.posterior_approximator.num_samples):
            self.state.put(state=self.state.get(i), i=i, keep=keep_top_n_checkpoints)
