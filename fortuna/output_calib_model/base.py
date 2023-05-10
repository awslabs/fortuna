import abc
import logging
from typing import Callable, Optional

import jax.numpy as jnp
from flax.core import FrozenDict

from fortuna.output_calib_model.config.base import Config
from fortuna.output_calib_model.loss import Loss
from fortuna.output_calib_model.output_calib_mixin import (
    WithOutputCalibCheckpointingMixin,
)
from fortuna.output_calib_model.output_calib_model_calibrator import (
    JittedOutputCalibModelCalibrator,
    MultiDeviceOutputCalibModelCalibrator,
    OutputCalibModelCalibrator,
)
from fortuna.output_calib_model.output_calib_state_repository import (
    OutputCalibStateRepository,
)
from fortuna.output_calib_model.state import OutputCalibState
from fortuna.output_calibrator.output_calib_manager.state import OutputCalibManagerState
from fortuna.typing import Array, Outputs, Path, Status, Targets
from fortuna.utils.device import select_trainer_given_devices
from fortuna.utils.random import RandomNumberGenerator


class OutputCalibModel(WithOutputCalibCheckpointingMixin, abc.ABC):
    """
    Abstract calibration model class.
    """

    def __init__(self, seed: int = 0):
        super().__init__()
        self.rng = RandomNumberGenerator(seed=seed)
        self.__set_rng()

    def __set_rng(self):
        self.output_calib_manager.rng = self.rng
        self.prob_output_layer.rng = self.rng
        self.predictive.rng = self.rng

    def _calibrate(
        self,
        uncertainty_fn: Callable[[jnp.ndarray, jnp.ndarray, Array], jnp.ndarray],
        loss_fn: Callable[[Outputs, Targets], jnp.ndarray],
        calib_outputs: Array,
        calib_targets: Array,
        val_outputs: Optional[Array] = None,
        val_targets: Optional[Array] = None,
        config: Config = Config(),
    ) -> Status:
        if (val_targets is not None and val_outputs is None) or (
            val_targets is None and val_outputs is not None
        ):
            raise ValueError(
                "For validation, both `val_outputs` and `val_targets` must be passed as arguments."
            )
        trainer_cls = select_trainer_given_devices(
            devices=config.processor.devices,
            BaseTrainer=OutputCalibModelCalibrator,
            JittedTrainer=JittedOutputCalibModelCalibrator,
            MultiDeviceTrainer=MultiDeviceOutputCalibModelCalibrator,
            disable_jit=config.processor.disable_jit,
        )

        calibrator = trainer_cls(
            calib_outputs=calib_outputs,
            calib_targets=calib_targets,
            val_outputs=val_outputs,
            val_targets=val_targets,
            predict_fn=self.prob_output_layer.predict,
            uncertainty_fn=uncertainty_fn,
            save_checkpoint_dir=config.checkpointer.save_checkpoint_dir,
            save_every_n_steps=config.checkpointer.save_every_n_steps,
            keep_top_n_checkpoints=config.checkpointer.keep_top_n_checkpoints,
            disable_training_metrics_computation=config.monitor.disable_calibration_metrics_computation,
            eval_every_n_epochs=config.monitor.eval_every_n_epochs,
            early_stopping_monitor=config.monitor.early_stopping_monitor,
            early_stopping_min_delta=config.monitor.early_stopping_min_delta,
            early_stopping_patience=config.monitor.early_stopping_patience,
        )

        if config.checkpointer.restore_checkpoint_path is None:
            state = OutputCalibManagerState.init_from_dict(
                d=FrozenDict(
                    output_calibrator=self.output_calib_manager.init(
                        output_dim=calib_outputs.shape[-1]
                    )
                ),
            )
            state = OutputCalibState.init(
                params=state.params,
                mutable=state.mutable,
                optimizer=config.optimizer.method,
            )
        else:
            state = self.restore_checkpoint(
                config.checkpointer.restore_checkpoint_path,
                optimizer=config.optimizer.method,
            )

        loss = Loss(self.predictive, loss_fn=loss_fn)
        loss.rng = self.rng

        if config.monitor.verbose:
            logging.info("Start calibration.")
        state, status = calibrator.train(
            rng=self.rng.get(),
            state=state,
            loss_fun=loss,
            n_epochs=config.optimizer.n_epochs,
            metrics=config.monitor.metrics,
            verbose=config.monitor.verbose,
        )

        self.predictive.state = OutputCalibStateRepository(
            config.checkpointer.save_checkpoint_dir
            if config.checkpointer.dump_state is True
            else None
        )
        self.predictive.state.put(
            state, keep=config.checkpointer.keep_top_n_checkpoints
        )
        return status

    def load_state(self, checkpoint_path: Path) -> None:
        """
        Load a calibration state from a checkpoint path.
        The checkpoint must be compatible with the calibration model.

        Parameters
        ----------
        checkpoint_path : Path
            Path to a checkpoint file or directory to restore.
        """
        try:
            self.restore_checkpoint(checkpoint_path)
        except ValueError:
            raise ValueError(
                f"No checkpoint was found in `checkpoint_path={checkpoint_path}`."
            )
        self.predictive.state = OutputCalibStateRepository(
            checkpoint_dir=checkpoint_path
        )

    def save_state(
        self, checkpoint_path: Path, keep_top_n_checkpoints: int = 1
    ) -> None:
        """
        Save the calibration state as a checkpoint.

        Parameters
        ----------
        checkpoint_path : Path
            Path to file or directory where to save the current state.
        keep_top_n_checkpoints : int
            Number of past checkpoint files to keep.
        """
        if self.predictive.state is None:
            raise ValueError(
                """No state available. You must first either calibrate the model, or load a saved checkpoint."""
            )
        return self.predictive.state.put(
            self.predictive.state.get(),
            checkpoint_path=checkpoint_path,
            keep=keep_top_n_checkpoints,
        )
