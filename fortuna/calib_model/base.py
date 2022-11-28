import abc
import logging
from typing import Callable, Optional

import jax.numpy as jnp
from flax.core import FrozenDict

from fortuna.calib_model.calib_config.base import CalibConfig
from fortuna.calib_model.calib_model_calibrator import (
    CalibModelCalibrator, JittedCalibModelCalibrator,
    MultiDeviceCalibModelCalibrator)
from fortuna.calibration.state import CalibState
from fortuna.output_calibrator.output_calib_manager.state import \
    OutputCalibManagerState
from fortuna.training.mixin import WithCheckpointingMixin
from fortuna.training.train_state_repository import TrainStateRepository
from fortuna.typing import Array, Path, Status
from fortuna.utils.device import select_trainer_given_devices
from fortuna.utils.random import RandomNumberGenerator


class CalibModel(WithCheckpointingMixin, abc.ABC):
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
        calib_outputs: Array,
        calib_targets: Array,
        val_outputs: Optional[Array] = None,
        val_targets: Optional[Array] = None,
        calib_config: CalibConfig = CalibConfig(),
    ) -> Status:
        if (val_targets is not None and val_outputs is None) or (
            val_targets is None and val_outputs is not None
        ):
            raise ValueError(
                "For validation, both `val_outputs` and `val_targets` must be passed as arguments."
            )
        trainer_cls = select_trainer_given_devices(
            devices=calib_config.processor.devices,
            BaseTrainer=CalibModelCalibrator,
            JittedTrainer=JittedCalibModelCalibrator,
            MultiDeviceTrainer=MultiDeviceCalibModelCalibrator,
            disable_jit=calib_config.processor.disable_jit,
        )

        calibrator = trainer_cls(
            calib_outputs=calib_outputs,
            calib_targets=calib_targets,
            val_outputs=val_outputs,
            val_targets=val_targets,
            predict_fn=self.prob_output_layer.predict,
            uncertainty_fn=uncertainty_fn,
            save_checkpoint_dir=calib_config.checkpointer.save_checkpoint_dir,
            save_every_n_steps=calib_config.checkpointer.save_every_n_steps,
            keep_top_n_checkpoints=calib_config.checkpointer.keep_top_n_checkpoints,
            disable_training_metrics_computation=calib_config.monitor.disable_calibration_metrics_computation,
            eval_every_n_epochs=calib_config.monitor.eval_every_n_epochs,
        )

        if calib_config.checkpointer.restore_checkpoint_path is None:
            state = OutputCalibManagerState.init_from_dict(
                d=FrozenDict(
                    output_calibrator=self.output_calib_manager.init(
                        output_dim=calib_outputs.shape[-1]
                    )
                ),
            )
            state = CalibState.init(
                params=state.params,
                mutable=state.mutable,
                optimizer=calib_config.optimizer.method,
            )
        else:
            state = self.restore_checkpoint(
                calib_config.checkpointer.restore_checkpoint_path,
                optimizer=calib_config.optimizer.method,
            )

        if calib_config.monitor.verbose:
            logging.info("Start calibration.")
        state, status = calibrator.train(
            rng=self.rng.get(),
            state=state,
            fun=self.predictive._log_joint_prob,
            n_epochs=calib_config.optimizer.n_epochs,
            metrics=calib_config.monitor.metrics,
            verbose=calib_config.monitor.verbose,
        )

        self.predictive.state = TrainStateRepository(
            calib_config.checkpointer.save_checkpoint_dir
            if calib_config.checkpointer.dump_state is True
            else None
        )
        self.predictive.state.put(
            state, keep=calib_config.checkpointer.keep_top_n_checkpoints
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
        self.predictive.state = TrainStateRepository(checkpoint_dir=checkpoint_path)

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
