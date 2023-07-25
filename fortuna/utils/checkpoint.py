from typing import Optional
from fortuna.typing import Path

from orbax.checkpoint import (
    Checkpointer,
    CheckpointManager,
    CheckpointManagerOptions,
    PyTreeCheckpointHandler,
)


def get_checkpoint_manager(
    checkpoint_dir: Path, keep_top_n_checkpoints: Optional[int] = None
):
    if checkpoint_dir is not None:
        options = CheckpointManagerOptions(
            create=True, max_to_keep=keep_top_n_checkpoints
        )
        return CheckpointManager(
            checkpoint_dir, Checkpointer(PyTreeCheckpointHandler()), options
        )
    return None
