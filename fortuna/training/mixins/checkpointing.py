import logging
from typing import Optional

from flax.training.orbax_utils import save_args_from_target
from jax import (
    ShapeDtypeStruct,
    local_devices,
)
from jax.sharding import SingleDeviceSharding
from jax.tree_util import (
    tree_map,
    tree_map_with_path,
)
from orbax.checkpoint import (
    ArrayRestoreArgs,
    CheckpointManager,
)

from fortuna.partitioner.partition_manager.base import PartitionManager
from fortuna.training.name_to_train_state import NameToTrainState
from fortuna.training.train_state import TrainState
from fortuna.typing import (
    OptaxOptimizer,
    Path,
)
from fortuna.utils.checkpoint import get_checkpoint_manager

logger = logging.getLogger(__name__)


class WithCheckpointingMixin:
    def __init__(
        self,
        *,
        partition_manager: Optional[PartitionManager] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        **kwargs,
    ):
        """
        Mixin class for all trainers that need checkpointing capabilities. This is a wrapper around functions in
        `flax.training.checkpoints.*`.

        Parameters
        ----------
        partition_manager: PartitionManager,
            An object that manages partitions.
        checkpoint_manager: CheckpointManager
            A checkpoint manager
        """
        super(WithCheckpointingMixin, self).__init__(**kwargs)
        self.partition_manager = partition_manager
        self.checkpoint_manager = checkpoint_manager

    def save_checkpoint(
        self,
        state: TrainState,
        save_checkpoint_dir: Path,
        keep: int = 1,
        force_save: bool = False,
    ) -> None:
        checkpoint_manager = (
            get_checkpoint_manager(
                checkpoint_dir=save_checkpoint_dir, keep_top_n_checkpoints=keep
            )
            if save_checkpoint_dir is not None
            else self.checkpoint_manager
        )
        if checkpoint_manager is not None:
            save_args = save_args_from_target(state)

            def save_ckpt_fn(_state):
                return checkpoint_manager.save(
                    _state.step,
                    _state,
                    force=force_save,
                    save_kwargs={"save_args": save_args},
                )

            if (
                hasattr(state, "grad_accumulated")
                and state.grad_accumulated is not None
            ):
                # do not save grad accumulated in the ckpt
                state = state.replace(grad_accumulated=None)
            save_ckpt_fn(state)

    def restore_checkpoint(
        self,
        restore_checkpoint_dir: Path,
        optimizer: Optional[OptaxOptimizer] = None,
        name_to_train_state: NameToTrainState = NameToTrainState,
    ) -> TrainState:
        ref = self._get_ref(lazy=False)
        restored = self.checkpoint_manager.restore(
            self.checkpoint_manager.latest_step(),
            items=ref,
            restore_kwargs={"restore_args": ref},
            directory=restore_checkpoint_dir,
        )
        if isinstance(restored, dict):
            name = "".join([chr(n) for n in restored["encoded_name"]])
            restored = name_to_train_state[name].value.init_from_dict(restored)

        if optimizer is not None:
            restored = restored.replace(
                tx=optimizer, opt_state=optimizer.init(restored.params)
            )

        return restored

    def get_shapes_dtypes_checkpoint(
        self,
        restore_checkpoint_dir: Path,
        name_to_train_state: NameToTrainState = NameToTrainState,
    ):
        ref = self._get_ref_without_shardings(lazy=True)
        state = self.checkpoint_manager.restore(
            self.checkpoint_manager.latest_step(),
            items=ref,
            restore_kwargs=dict(restore_args=ref),
            directory=restore_checkpoint_dir,
        )
        name = "".join([chr(n) for n in state["encoded_name"].get().tolist()])
        state = name_to_train_state[name].value.init_from_dict(state)
        return tree_map(lambda v: _get_shapes_dtypes(v.get()), state)

    def _get_ref_from_shardings(self):
        return tree_map_with_path(
            lambda p, sharding, shape_dtype: ArrayRestoreArgsWithShape(
                mesh=self.partition_manager.partitioner.mesh,
                sharding=sharding,
                dtype=shape_dtype.dtype,
                shape=shape_dtype.shape,
            ),
            self.partition_manager.shardings,
            self.partition_manager.shapes_dtypes,
        )

    def _get_ref_without_shardings(self, lazy):
        return tree_map_with_path(
            lambda p, v: ArrayRestoreArgs(
                lazy=lazy, sharding=SingleDeviceSharding(device=local_devices()[0])
            ),
            self.checkpoint_manager.structure(),
        )

    def _get_ref(self, lazy=False):
        if (
            self.partition_manager is not None
            and self.partition_manager.shardings is not None
            and self.partition_manager.shapes_dtypes is not None
        ):
            return self._get_ref_from_shardings()
        return self._get_ref_without_shardings(lazy=False)


class ArrayRestoreArgsWithShape(ArrayRestoreArgs):
    def __init__(self, shape, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shape = shape


def _get_shapes_dtypes(v):
    return ShapeDtypeStruct(shape=v.shape, dtype=v.dtype)
