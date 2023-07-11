import logging
from typing import (
    Dict,
    Optional,
)

from flax.training.early_stopping import EarlyStopping

logger = logging.getLogger(__name__)


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
            self._early_stopping = EarlyStopping(
                min_delta=early_stopping_min_delta, patience=early_stopping_patience
            )
            if early_stopping_verbose:
                logging.info(
                    "If validation data are provided, early stopping will be enabled."
                )

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
            improved, self._early_stopping = self._early_stopping.update(
                early_stopping_monitor
            )
        return improved
