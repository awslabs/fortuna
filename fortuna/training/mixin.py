import logging
import os
from typing import Dict, Optional

from flax.training import checkpoints
from flax.training.early_stopping import EarlyStopping
from fortuna.training.name_to_train_state import NameToTrainState
from fortuna.training.train_state import TrainState
from fortuna.typing import OptaxOptimizer, Path

logger = logging.getLogger(__name__)


class WithCheckpointingMixin:
    def __init__(
        self, **kwargs,
    ):
        """
        Mixin class for all trainers that need checkpointing capabilities. This is a wrapper around functions in
        `flax.training.checkpoints.*`.
        """
        super(WithCheckpointingMixin, self).__init__(**kwargs)

    def save_checkpoint(
        self,
        state: TrainState,
        save_checkpoint_dir: Path,
        keep: int = 1,
        force_save: bool = False,
        prefix: str = "checkpoint_",
    ) -> None:
        if save_checkpoint_dir:
            checkpoints.save_checkpoint(
                ckpt_dir=str(save_checkpoint_dir),
                target=state,
                step=state.step,
                prefix=prefix,
                keep=keep,
                overwrite=force_save,
            )

    def restore_checkpoint(
        self,
        restore_checkpoint_path: Path,
        optimizer: Optional[OptaxOptimizer] = None,
        prefix: str = "checkpoint_",
        name_to_train_state: NameToTrainState = NameToTrainState,
        **kwargs,
    ) -> TrainState:
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
        name = "".join([chr(n) for n in d["encoded_name"].tolist()])
        return name_to_train_state[name].value.init_from_dict(d, optimizer, **kwargs)

    def get_path_latest_checkpoint(
        self, checkpoint_dir: Path, prefix: str = "checkpoint_"
    ) -> Optional[str]:
        return checkpoints.latest_checkpoint(ckpt_dir=checkpoint_dir, prefix=prefix)


class WithEarlyStoppingMixin:
    def __init__(
        self,
        *,
        early_stopping_monitor: str = "val_loss",
        early_stopping_min_delta: float = 0.0,
        early_stopping_patience: Optional[int] = 0,
        early_stopping_mode: str = "min",
        early_stopping_verbose: bool = True,
        **kwargs,
    ):
        super(WithEarlyStoppingMixin, self).__init__(**kwargs)
        self.early_stopping_monitor = early_stopping_monitor
        self.early_stopping_mode = early_stopping_mode
        self.early_stopping_patience = early_stopping_patience

        if early_stopping_patience is None or early_stopping_patience <= 0:
            if early_stopping_verbose:
                logging.info(
                    f"Early stopping not enabled. Set `early_stopping_patience>=0` to enable it."
                )
        elif self.early_stopping_mode is None or self.early_stopping_mode not in (
            "min",
            "max",
        ):
            if early_stopping_verbose:
                logging.warning(
                    f"`early_stopping_mode={early_stopping_mode}` is not a valid. Early stopping will be disabled."
                )
        else:
            self.early_stopping = EarlyStopping(
                min_delta=early_stopping_min_delta, patience=early_stopping_patience
            )
            if early_stopping_verbose:
                logging.info("Early Stopping is enabled. ")

    @property
    def is_early_stopping_active(self) -> bool:
        return not (
            (self.early_stopping_patience is None or self.early_stopping_patience <= 0)
            or (
                self.early_stopping_mode is None
                or self.early_stopping_mode not in ("min", "max")
            )
        )

    def early_stopping_update(
        self, validation_metrics: Dict[str, float]
    ) -> Optional[bool]:
        improved = None
        if self.is_early_stopping_active:
            early_stopping_monitor = validation_metrics[self.early_stopping_monitor]
            if self.early_stopping_mode == "max":
                early_stopping_monitor = -early_stopping_monitor
            improved, self.early_stopping = self.early_stopping.update(
                early_stopping_monitor
            )
        return improved


class InputValidatorMixin:
    def __init__(self, *args, **kwargs):
        if len(args) > 0:
            raise AttributeError("Cannot recognize inputs arguments: {}".format(args))
        if len(kwargs) > 0:
            raise AttributeError(
                "{} are not valid input arguments.".format(list(kwargs.keys()))
            )
        super(InputValidatorMixin, self).__init__(*args, **kwargs)
