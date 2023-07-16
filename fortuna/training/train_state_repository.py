from shutil import rmtree
from typing import (
    Dict,
    List,
    Optional,
    Union,
)

from jax import eval_shape
from orbax.checkpoint import CheckpointManager

from fortuna.partitioner.partition_manager.base import PartitionManager
from fortuna.training.mixins.checkpointing import WithCheckpointingMixin
from fortuna.training.train_state import TrainState
from fortuna.typing import (
    OptaxOptimizer,
    Path,
)


class TrainStateRepository(WithCheckpointingMixin):
    def __init__(
        self,
        partition_manager: Optional[PartitionManager] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
    ):
        super().__init__(partition_manager=partition_manager)
        self.checkpoint_manager = checkpoint_manager
        self._state = None

    def get(
        self,
        checkpoint_dir: Optional[Path] = None,
        optimizer: Optional[OptaxOptimizer] = None,
        _do_reshard: bool = True
    ) -> Union[Dict, TrainState]:
        if not checkpoint_dir and not self.checkpoint_manager and not self._state:
            raise ValueError("No state available.")
        if checkpoint_dir or self.checkpoint_manager:
            return self.restore_checkpoint(
                restore_checkpoint_dir=checkpoint_dir, optimizer=optimizer
            )
        if optimizer is not None:
            if self.partition_manager is not None and _do_reshard:
                state = self.partition_manager.reshard(self._state)
                return state.replace(tx=optimizer, opt_state=optimizer.init(state.params))
            else:
                self._state = self._state.replace(tx=optimizer, opt_state=optimizer.init(self._state.params))
        return self._state

    def put(
        self,
        state: TrainState,
        checkpoint_dir: Optional[Path] = None,
        keep: int = 1,
    ) -> None:
        if checkpoint_dir or self.checkpoint_manager:
            self.save_checkpoint(
                state=state,
                save_checkpoint_dir=checkpoint_dir,
                keep=keep,
                force_save=True,
            )
        else:
            self._state = state

    def pull(
        self,
        checkpoint_dir: Path = None,
        optimizer: Optional[OptaxOptimizer] = None,
    ) -> TrainState:
        state = self.get(
            checkpoint_dir=checkpoint_dir,
            optimizer=optimizer,
        )
        if checkpoint_dir or self.checkpoint_manager:
            if checkpoint_dir is None:
                self.checkpoint_manager.delete(self.checkpoint_manager.latest_step())
            else:
                rmtree(checkpoint_dir)
        return state

    def update(
        self,
        variables: Dict,
        checkpoint_dir: Path = None,
        optimizer: Optional[OptaxOptimizer] = None,
        keep: int = 1,
    ):
        state = self.pull(
            checkpoint_dir=checkpoint_dir,
            optimizer=optimizer,
        )
        state = state.replace(**variables)
        self.put(state, checkpoint_dir=checkpoint_dir, keep=keep)

    def extract(self, keys: List[str], checkpoint_dir: Optional[Path] = None) -> Dict:
        state = self.get(checkpoint_dir=checkpoint_dir)
        return {k: getattr(state, k) for k in keys}
