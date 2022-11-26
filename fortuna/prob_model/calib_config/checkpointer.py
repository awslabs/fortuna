from typing import Optional

from fortuna.typing import Path


class CalibCheckpointer:
    def __init__(
        self,
        save_checkpoint_dir: Optional[Path] = None,
        restore_checkpoint_path: Optional[Path] = None,
        save_every_n_steps: Optional[int] = None,
        keep_top_n_checkpoints: Optional[int] = 2,
        dump_state: bool = False,
    ):
        """
        An object to configure saving and restoring of checkpoints during the calibration process.

        Parameters
        ----------
        save_checkpoint_dir: Optional[Path] = None
            Save directory location.
        restore_checkpoint_path: Optional[Path]
            Path to checkpoint file or directory to restore.
        save_every_n_steps: int
            Number of training steps between checkpoints. To disable, set `every_n_train_steps` to None or 0 (no
            checkpoint will be saved during training).
        keep_top_n_checkpoints: int
            Number of past checkpoint files to keep.
        dump_state: bool
            Dump the fitted calibration state as a checkpoint in `save_checkpoint_dir`.
            Any future call to the state will internally involve restoring it from memory.
        """
        self.save_checkpoint_dir = save_checkpoint_dir
        self.save_every_n_steps = save_every_n_steps
        self.restore_checkpoint_path = restore_checkpoint_path
        self.keep_top_n_checkpoints = keep_top_n_checkpoints
        self.dump_state = dump_state
