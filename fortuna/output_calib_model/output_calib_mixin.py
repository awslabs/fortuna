from fortuna.output_calib_model.state import OutputCalibState
from fortuna.training.mixin import WithCheckpointingMixin
from fortuna.typing import Path, OptaxOptimizer
from typing import Optional
import os
from flax.training import checkpoints


class WithOutputCalibCheckpointingMixin(WithCheckpointingMixin):
    def restore_checkpoint(
        self,
        restore_checkpoint_dir: Path,
        optimizer: Optional[OptaxOptimizer] = None,
        prefix: str = "checkpoint_",
        **kwargs,
    ) -> OutputCalibState:
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

        return OutputCalibState.init_from_dict(d, optimizer, **kwargs)
