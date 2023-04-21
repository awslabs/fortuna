import abc
import logging
from typing import Callable, Optional, Tuple

import jax.numpy as jnp

from fortuna.data.loader import DataLoader
from fortuna.typing import Array, Path, Status
from fortuna.utils.device import select_trainer_given_devices
from fortuna.utils.random import RandomNumberGenerator
from fortuna.calibration.finetune_calib_model.finetune_calib_state_repository import FinetuneCalibStateRepository
from fortuna.calibration.finetune_calib_model.finetune_calib_mixin import WithFinetuneCalibCheckpointingMixin
from fortuna.calibration.finetune_calib_model.finetune_calib_model_calibrator import FinetuneCalibModelCalibrator, \
    JittedFinetuneCalibModelCalibrator, MultiDeviceFinetuneCalibModelCalibrator
from fortuna.calibration.finetune_calib_model.config.base import Config
from fortuna.calibration.loss.base import Loss
from fortuna.model.model_manager.state import ModelManagerState


class FinetuneCalibModel(WithFinetuneCalibCheckpointingMixin, abc.ABC):
    """
    A fine-tuning calibration model.
    """

    def __init__(
            self,
            seed: int = 0
    ):
        super().__init__()
        self.rng = RandomNumberGenerator(seed=seed)
        self.__set_rng()

    def __set_rng(self):
        self.model_manager.rng = self.rng
        self.prob_output_layer.rng = self.rng
        self.predictive.rng = self.rng

    def _calibrate(
        self,
        calib_data_loader: DataLoader,
        uncertainty_fn: Callable[[jnp.ndarray, jnp.ndarray, Array], jnp.ndarray],
        val_data_loader: Optional[DataLoader] = None,
        loss_fn: Optional[Loss] = None,
        config: Config = Config()
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
            BaseTrainer=FinetuneCalibModelCalibrator,
            JittedTrainer=JittedFinetuneCalibModelCalibrator,
            MultiDeviceTrainer=MultiDeviceFinetuneCalibModelCalibrator,
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

        state = self.restore_checkpoint(
            restore_checkpoint_path=config.checkpointer.restore_checkpoint_path,
            optimizer=config.optimizer.method,
        ) if config.checkpointer.restore_checkpoint_path else self.predictive.state.get(
            optimizer=config.optimizer.method
        )

        if loss_fn is not None:
            def fun(p, t, o, m, r, a):
                return -loss_fn(p, t, o, m, r, a)
        else:
            fun = self.predictive.likelihood._batched_log_joint_prob

        n_calib_data = calib_data_loader.size
        n_val_data = val_data_loader.size if val_data_loader is not None else None

        if config.monitor.verbose:
            logging.info("Start calibration.")

        state, status = trainer.train(
            rng=self.rng.get(),
            state=state,
            fun=fun,
            training_dataloader=calib_data_loader,
            training_dataset_size=n_calib_data,
            n_epochs=config.optimizer.n_epochs,
            metrics=config.monitor.metrics,
            validation_dataloader=val_data_loader,
            validation_dataset_size=n_val_data,
            verbose=config.monitor.verbose,
            callbacks=config.callbacks,
        )
        self.state = FinetuneCalibStateRepository(
            config.checkpointer.save_checkpoint_dir
            if config.checkpointer.dump_state is True
            else None
        )
        self.state.put(state, keep=config.checkpointer.keep_top_n_checkpoints)
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
        self.predictive.state = FinetuneCalibStateRepository(checkpoint_dir=checkpoint_path)

    def save_state(
        self, checkpoint_path: Path, keep_top_n_checkpoints: int = 1
    ) -> None:
        return self.predictive.state.put(
            self.predictive.state.get(),
            checkpoint_path=checkpoint_path,
            keep=keep_top_n_checkpoints,
        )

    def _get_output_dim(self, input_shape: Tuple, **kwargs) -> int:
        """
        Initialize the state of the joint distribution.

        Parameters
        ----------
        input_shape : Tuple
            The shape of the input variable.

        Returns
        -------
        A state of the joint distribution.
        """
        oms = ModelManagerState.init_from_dict(
            self.model_manager.init(
                input_shape, rng=self.rng.get(), **kwargs
            )
        )
        return self.model_manager.apply(
            oms.params, jnp.zeros((1,) + input_shape), mutable=oms.mutable
        ).shape[-1]



