import os
from typing import Optional

from flax.training import checkpoints

from fortuna.calib_model.state import CalibState
from fortuna.training.mixin import WithCheckpointingMixin
from fortuna.typing import OptaxOptimizer, Path


class WithCalibCheckpointingMixin(WithCheckpointingMixin):
    def restore_checkpoint(
        self,
        restore_checkpoint_path: Path,
        optimizer: Optional[OptaxOptimizer] = None,
        prefix: str = "checkpoint_",
        **kwargs,
    ) -> CalibState:
        if not os.path.isdir(restore_checkpoint_path) and not os.path.isfile(
            restore_checkpoint_path
        ):
            raise ValueError(
                f"`restore_checkpoint_path={restore_checkpoint_path}` was not found."
            )
        d = checkpoints.restore_checkpoint(
            ckpt_dir=str(restore_checkpoint_path),
            target=None,
            step=None,
            prefix=prefix,
            parallel=True,
        )
        if d is None:
            raise ValueError(
                f"No checkpoint was found in `restore_checkpoint_path={restore_checkpoint_path}`."
            )

        return CalibState.init_from_dict(d, optimizer, **kwargs)
