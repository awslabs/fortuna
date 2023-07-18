from typing import Optional

from fortuna.typing import Path


class FitCheckpointer:
    def __init__(
        self,
        save_checkpoint_dir: Optional[Path] = None,
        restore_checkpoint_dir: Optional[Path] = None,
        start_from_current_state: bool = False,
        save_every_n_steps: Optional[int] = None,
        keep_top_n_checkpoints: Optional[int] = 2,
        dump_state: bool = False,
        checkpoint_type: str = "last",
    ):
        """
        An object to configure saving and restoring of checkpoints during the posterior fitting.

        Parameters
        ----------
        save_checkpoint_dir: Optional[Path]
            Save directory location.
        restore_checkpoint_dir: Optional[Path]
            Path to checkpoint file or directory to restore.
        start_from_current_state: bool = False
            If True, the optimization will start from the current state. If `restore_checkpoint_dir` is given, then
            `start_from_current_state` is ignored.
        save_every_n_steps: int
            Number of training steps between checkpoints. To disable, set `every_n_train_steps` to None or 0 (no
            checkpoint will be saved during training).
        keep_top_n_checkpoints: int
            Number of past checkpoint files to keep.
        dump_state: bool
            Dump the fitted posterior state as a checkpoint in `save_checkpoint_dir`. Any future call to the state will
            internally involve restoring it from memory.
        checkpoint_type: str
            Which checkpoint type to pass to the state.
            There are two possible options:

            - "last": this is the state obtained at the end of training.
            - "best": this is the best checkpoint with respect to the metric monitored by early stopping. Notice that
              this might be available only if validation data is provided, and both checkpoint saving and early
              stopping are enabled.
        """
        self.save_checkpoint_dir = save_checkpoint_dir
        self.save_every_n_steps = save_every_n_steps
        self.restore_checkpoint_dir = restore_checkpoint_dir
        self.start_from_current_state = start_from_current_state
        self.keep_top_n_checkpoints = keep_top_n_checkpoints
        self.dump_state = dump_state

        allowed_checkpoint_types = ["last", "best"]
        if checkpoint_type not in allowed_checkpoint_types:
            raise ValueError(
                f"`checkpoint_type={checkpoint_type}` not recognised. "
                f"Pleas select one of the following options: {allowed_checkpoint_types}."
            )
        self.checkpoint_type = checkpoint_type
