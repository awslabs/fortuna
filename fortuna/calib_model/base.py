import abc
import logging
from typing import (
    Callable,
    Optional,
    Tuple,
    Type
)

from flax.core import FrozenDict
import jax.numpy as jnp
from jax import eval_shape
from fortuna.partitioner.partition_manager.base import PartitionManager
from fortuna.calib_model.calib_model_calibrator import (
    CalibModelCalibrator,
    ShardedCalibModelCalibrator,
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
    Shape,
    Status,
    Targets,
    Uncertainties,
)
import pathlib
from jax._src.prng import PRNGKeyArray
from orbax.checkpoint import CheckpointManager
from fortuna.utils.checkpoint import get_checkpoint_manager
from fortuna.utils.data import get_inputs_from_shape
from fortuna.utils.freeze import get_trainable_paths
from fortuna.utils.nested_dicts import (
    nested_get,
    nested_set,
)
from fortuna.utils.random import RandomNumberGenerator


class CalibModel(abc.ABC):
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

        trainer_cls = (
            ShardedCalibModelCalibrator if not config.processor.disable_jit else CalibModelCalibrator
        )

        trainer = trainer_cls(
            predict_fn=self.prob_output_layer.predict,
            partition_manager=self.partition_manager,
            checkpoint_manager=get_checkpoint_manager(
                config.checkpointer.save_checkpoint_dir,
                keep_top_n_checkpoints=config.checkpointer.keep_top_n_checkpoints,
            ),
            uncertainty_fn=uncertainty_fn,
            save_checkpoint_dir=config.checkpointer.save_checkpoint_dir,
            save_every_n_steps=config.checkpointer.save_every_n_steps,
            keep_top_n_checkpoints=config.checkpointer.keep_top_n_checkpoints,
            disable_training_metrics_computation=config.monitor.disable_calibration_metrics_computation,
            eval_every_n_epochs=config.monitor.eval_every_n_epochs,
            early_stopping_monitor=config.monitor.early_stopping_monitor,
            early_stopping_min_delta=config.monitor.early_stopping_min_delta,
            early_stopping_patience=config.monitor.early_stopping_patience,
            freeze_fun=config.optimizer.freeze_fun,
        )

        checkpoint_restorer = (
            get_checkpoint_manager(
                str(
                    pathlib.Path(config.checkpointer.restore_checkpoint_dir)
                    / config.checkpointer.checkpoint_type
                ),
                keep_top_n_checkpoints=config.checkpointer.keep_top_n_checkpoints,
            )
            if config.checkpointer.restore_checkpoint_dir is not None
            else None
        )

        if self._is_state_available_somewhere(config):
            state = self._restore_state_from_somewhere(
                config=config,
                allowed_states=(CalibState,),
                checkpoint_manager=checkpoint_restorer,
                partition_manager=self.partition_manager,
            )
            state = self._freeze_optimizer_in_state(state, config)
            self.partition_manager.shapes_dtypes = eval_shape(lambda: state)
        else:
            input_shape = calib_data_loader.input_shape

            def init_state_fn(rng):
                _state = self._init_state(
                    input_shape=input_shape, config=config, rng=rng
                )
                return self._freeze_optimizer_in_state(_state, config)

            state = self.partition_manager.init_sharded_state(
                init_state_fn, self.rng.get()
            )

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
            training_data_loader=calib_data_loader,
            training_dataset_size=n_calib_data,
            n_epochs=config.optimizer.n_epochs,
            metrics=config.monitor.metrics,
            validation_data_loader=val_data_loader,
            validation_dataset_size=n_val_data,
            verbose=config.monitor.verbose,
            callbacks=config.callbacks,
        )

        self.predictive.state = CalibStateRepository(
            partition_manager=self.partition_manager,
            checkpoint_manager=get_checkpoint_manager(
                checkpoint_dir=str(
                    pathlib.Path(config.checkpointer.save_checkpoint_dir)
                    / config.checkpointer.checkpoint_type
                ),
                keep_top_n_checkpoints=config.checkpointer.keep_top_n_checkpoints,
            )
            if config.checkpointer.save_checkpoint_dir is not None
            and config.checkpointer.dump_state
            else None,
        )
        if self.predictive.state.checkpoint_manager is None:
            self.predictive.state.put(state, keep=config.checkpointer.keep_top_n_checkpoints)

        if config.monitor.verbose:
            logging.info("Calibration completed.")
        return status

    def load_state(self, checkpoint_dir: Path) -> None:
        """
        Load the state of the posterior distribution from a checkpoint path. The checkpoint must be compatible with the
        probabilistic model.

        Parameters
        ----------
        checkpoint_dir : Path
            Path to a checkpoint file or directory to restore.
        """
        self.predictive.state = CalibStateRepository(
            partition_manager=self.partition_manager,
            checkpoint_manager=get_checkpoint_manager(checkpoint_dir=checkpoint_dir),
        )
        self.partition_manager.shapes_dtypes = self.predictive.state.get_shapes_dtypes_checkpoint()

    def save_state(self, checkpoint_dir: Path, keep_top_n_checkpoints: int = 1) -> None:
        """
        Save the state of the calibration model to a checkpoint directory.

        Parameters
        ----------
        checkpoint_dir: Path
            Path to checkpoint file or directory to restore.
        keep_top_n_checkpoints: int
            Number of past checkpoint files to keep.
        """
        if self.predictive.state is None:
            raise ValueError(
                """No state available. You must first either fit the posterior distribution, or load a
            saved checkpoint."""
            )
        return self.predictive.state.put(
            self.predictive.state.get(),
            checkpoint_dir=checkpoint_dir,
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
        inputs = get_inputs_from_shape(input_shape)
        outputs = self.model_manager.apply(oms.params, inputs, mutable=oms.mutable)
        return (
            outputs[0].shape[-1]
            if isinstance(outputs, (list, tuple))
            else outputs.shape[-1]
        )

    def _init_state(self, input_shape: Shape, config: Config, rng: Optional[PRNGKeyArray] = None):
        if rng is None:
            rng = self.rng.get()

        state = ModelManagerState.init_from_dict(
            self.likelihood.model_manager.init(input_shape, rng=rng)
        )
        return CalibState.init(
            params=state.params,
            mutable=state.mutable,
            optimizer=config.optimizer.method,
        )

    def _freeze_optimizer_in_state(
            self,
            state: CalibState,
            config: Config
    ) -> CalibState:
        if config.optimizer.freeze_fun is not None:
            trainable_paths = get_trainable_paths(
                state.params, config.optimizer.freeze_fun
            )
            state = state.replace(
                opt_state=config.optimizer.method.init(
                    FrozenDict(
                        nested_set(
                            d={},
                            key_paths=trainable_paths,
                            objs=tuple(
                                [
                                    nested_get(state.params.unfreeze(), path)
                                    for path in trainable_paths
                                ]
                            ),
                            allow_nonexistent=True,
                        )
                    )
                )
            )
        return state

    def _is_state_available_somewhere(self, config: Config) -> bool:
        return (
            config.checkpointer.restore_checkpoint_dir is not None
            or config.checkpointer.start_from_current_state
        )

    def _restore_state_from_somewhere(
        self,
        config: Config,
        allowed_states: Optional[Tuple[Type[CalibState], ...]] = None,
        partition_manager: Optional[PartitionManager] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
    ) -> CalibState:
        if checkpoint_manager is not None:
            repo = CalibStateRepository(
                partition_manager=partition_manager,
                checkpoint_manager=checkpoint_manager,
            )
            state = repo.get(optimizer=config.optimizer.method)
        elif config.checkpointer.start_from_current_state:
            state = self.predictive.state.get(optimizer=config.optimizer.method)

        if allowed_states is not None and not isinstance(state, allowed_states):
            raise ValueError(
                f"The type of the restored checkpoint must be within {allowed_states}. "
                f"However, the restored checkpoint has type {type(state)}."
            )

        return state
