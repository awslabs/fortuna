from fortuna.calibration.finetune_calib_model.state import FinetuneCalibState
from fortuna.training.mixin import WithCheckpointingMixin
from fortuna.typing import Path, OptaxOptimizer
from typing import Optional
import os
from flax.training import checkpoints


class WithFinetuneCalibCheckpointingMixin(WithCheckpointingMixin):
    def restore_checkpoint(
        self,
        restore_checkpoint_path: Path,
        optimizer: Optional[OptaxOptimizer] = None,
        prefix: str = "checkpoint_",
        **kwargs,
    ) -> FinetuneCalibState:
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

        return FinetuneCalibState.init_from_dict(d, optimizer, **kwargs)
