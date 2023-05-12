import abc
import logging
from typing import (
    Callable,
    Optional,
    Tuple,
)

from flax.core.frozen_dict import freeze
from flax.traverse_util import path_aware_map
import jax.numpy as jnp
import optax

from fortuna.calib_model.calib_mixin import WithCalibCheckpointingMixin
from fortuna.calib_model.calib_model_calibrator import (
    CalibModelCalibrator,
    JittedCalibModelCalibrator,
    MultiDeviceCalibModelCalibrator,
)
from fortuna.calib_model.calib_state_repository import CalibStateRepository
from fortuna.calib_model.config.base import Config
from fortuna.calib_model.loss import Loss
from fortuna.calib_model.state import CalibState
from fortuna.data.loader import DataLoader
from fortuna.model.model_manager.state import ModelManagerState
from fortuna.typing import (
    Outputs,
    Path,
    Predictions,
    Status,
    Targets,
    Uncertainties,
    Shape,
)
from fortuna.utils.data import get_input_shape
from fortuna.utils.device import select_trainer_given_devices
from fortuna.utils.random import RandomNumberGenerator


class CalibModel(WithCalibCheckpointingMixin, abc.ABC):
    def __init__(self, seed: int = 0):
        """
        A calibration model.

        Parameters
        ----------
        seed: int = 0
            Random seed.
        """
        super().__init__()
        self.rng = RandomNumberGenerator(seed=seed)
        self.__set_rng()

    def __set_rng(self):
        self.model_manager.rng = self.rng
        self.prob_output_layer.rng = self.rng
        self.likelihood.rng = self.rng
        self.predictive.rng = self.rng

    def _calibrate(
        self,
        calib_data_loader: DataLoader,
        uncertainty_fn: Callable[[Predictions, Uncertainties, Targets], jnp.ndarray],
        loss_fn: Callable[[Outputs, Targets], jnp.ndarray],
        val_data_loader: Optional[DataLoader] = None,
        config: Config = Config(),
    ) -> Status:
        if (
            config.checkpointer.dump_state is True
            and not config.checkpointer.save_checkpoint_dir
        ):
            raise ValueError(
                "`save_checkpoint_dir` must be passed when `dump_state` is set to True."
            )

        trainer_cls = select_trainer_given_devices(
            devices=config.processor.devices,
            base_trainer_cls=CalibModelCalibrator,
            jitted_trainer_cls=JittedCalibModelCalibrator,
            multi_device_trainer_cls=MultiDeviceCalibModelCalibrator,
            disable_jit=config.processor.disable_jit,
        )

        trainer = trainer_cls(
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

        state = self._init_state(calib_data_loader, config)

        if config.optimizer.freeze_fun is not None:
            partition_optimizers = {
                "trainable": config.optimizer.method,
                "frozen": optax.set_to_zero(),
            }
            partition_params = freeze(
                path_aware_map(config.optimizer.freeze_fun, state.params)
            )
            config.optimizer.method = optax.multi_transform(
                partition_optimizers, partition_params
            )
            state = self._init_state(calib_data_loader, config)

        loss = Loss(self.likelihood, loss_fn=loss_fn)
        loss.rng = self.rng

        n_calib_data = calib_data_loader.size
        n_val_data = val_data_loader.size if val_data_loader is not None else None

        if config.monitor.verbose:
            logging.info("Start calibration.")

        state, status = trainer.train(
            rng=self.rng.get(),
            state=state,
            loss_fun=loss,
            training_dataloader=calib_data_loader,
            training_dataset_size=n_calib_data,
            n_epochs=config.optimizer.n_epochs,
            metrics=config.monitor.metrics,
            validation_dataloader=val_data_loader,
            validation_dataset_size=n_val_data,
            verbose=config.monitor.verbose,
            callbacks=config.callbacks,
        )
        self.predictive.state = CalibStateRepository(
            config.checkpointer.save_checkpoint_dir
            if config.checkpointer.dump_state is True
            else None
        )
        self.predictive.state.put(
            state, keep=config.checkpointer.keep_top_n_checkpoints
        )
        if config.monitor.verbose:
            logging.info("Calibration completed.")
        return status

    def load_state(self, checkpoint_path: Path) -> None:
        """
        Load the state of the posterior distribution from a checkpoint path. The checkpoint must be compatible with the
        probabilistic model.

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
        self.predictive.state = CalibStateRepository(checkpoint_dir=checkpoint_path)

    def save_state(
        self, checkpoint_path: Path, keep_top_n_checkpoints: int = 1
    ) -> None:
        return self.predictive.state.put(
            self.predictive.state.get(),
            checkpoint_path=checkpoint_path,
            keep=keep_top_n_checkpoints,
        )

    def _get_output_dim(self, input_shape: Shape, **kwargs) -> int:
        """
        Initialize the state of the joint distribution.

        Parameters
        ----------
        input_shape : Shape
            The shape of the input variable.

        Returns
        -------
        A state of the joint distribution.
        """
        oms = ModelManagerState.init_from_dict(
            self.model_manager.init(input_shape, rng=self.rng.get(), **kwargs)
        )
        outputs = self.model_manager.apply(
            oms.params, jnp.zeros((1,) + input_shape), mutable=oms.mutable
        )
        return outputs[0].shape[-1] if isinstance(outputs, (list, tuple)) else outputs.shape[-1]

    def _init(self, data_loader: DataLoader, config: Config):
        for inputs, targets in data_loader:
            input_shape = get_input_shape(inputs)
            break

        state = ModelManagerState.init_from_dict(
            self.likelihood.model_manager.init(input_shape, rng=self.rng.get())
        )
        return CalibState.init(
            params=state.params,
            mutable=state.mutable,
            optimizer=config.optimizer.method,
        )

    def _init_state(self, calib_data_loader: DataLoader, config: Config) -> CalibState:
        if config.checkpointer.restore_checkpoint_path is None:
            if config.checkpointer.start_from_current_state:
                state = self.predictive.state.get(optimizer=config.optimizer.method)
            else:
                state = self._init(calib_data_loader, config)
        else:
            if config.checkpointer.start_from_current_state:
                logging.warning(
                    "`config.checkpointer.start_from_current_state` will be ignored since "
                    "`config.checkpointer.restore_checkpoint_path` is given."
                )
            state = self.restore_checkpoint(
                restore_checkpoint_path=config.checkpointer.restore_checkpoint_path,
                optimizer=config.optimizer.method,
            )
        return state
