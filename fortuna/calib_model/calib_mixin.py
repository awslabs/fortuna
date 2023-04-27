from fortuna.calib_model.state import CalibState
from fortuna.training.mixin import WithCheckpointingMixin
from fortuna.typing import Path, OptaxOptimizer
from typing import Optional
import os
from flax.training import checkpoints


class WithCalibCheckpointingMixin(WithCheckpointingMixin):
    def restore_checkpoint(
        self,
        restore_checkpoint_dir: Path,
        optimizer: Optional[OptaxOptimizer] = None,
        prefix: str = "checkpoint_",
        **kwargs,
    ) -> CalibState:
        if not os.path.isdir(restore_checkpoint_dir) and not os.path.isfile(
            restore_checkpoint_dir
        ):
            raise ValueError(
                f"`restore_checkpoint_dir={restore_checkpoint_dir}` was not found."
            )
        d = checkpoints.restore_checkpoint(
            ckpt_dir=str(restore_checkpoint_dir),
            target=None,
            step=None,
            prefix=prefix,
            parallel=True,
        )
        if d is None:
            raise ValueError(
                f"No checkpoint was found in `restore_checkpoint_dir={restore_checkpoint_dir}`."
            )

        return CalibState.init_from_dict(d, optimizer, **kwargs)
