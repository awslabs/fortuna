from typing import Optional

from fortuna.typing import Path


class CalibCheckpointer:
    def __init__(
        self,
        save_state_path: Optional[Path] = None,
        keep_top_n_checkpoints: Optional[int] = 2,
        start_from_current_state: bool = False,
        restore_checkpoint_path: Optional[Path] = None,
    ):
        """
        An object to configure saving and restoring of checkpoints during the calibration process.

        Parameters
        ----------
        save_state_path: Optional[Path] = None
            Filepath or directory where to save the calibrated state.
        keep_top_n_checkpoints: int
            Number of past checkpoint files to keep.
        start_from_current_state: bool
            Start the calibration from the current state.
        restore_checkpoint_path: Optional[Path]
            Filepath or directory to a checkpoint to restore and start the calibration from.
        """
        if start_from_current_state and restore_checkpoint_path is not None:
            raise ValueError(
                "Please set `start_from_current_state` to True or pass `restore_checkpoint_path` "
                "to `CalibCheckpointer`, not both."
            )
        self.save_state_path = save_state_path
        self.keep_top_n_checkpoints = keep_top_n_checkpoints
        self.start_from_current_state = start_from_current_state
        self.restore_checkpoint_path = restore_checkpoint_path
