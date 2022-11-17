from typing import Callable, Optional, Tuple

import jax.numpy as jnp


class FitMonitor:
    def __init__(
        self,
        metrics: Optional[Tuple[Callable[[jnp.ndarray], float], ...]] = None,
        early_stopping_patience: int = 0,
        early_stopping_monitor: str = "val_loss",
        early_stopping_min_delta: float = 0.0,
        eval_every_n_epochs: int = 1,
        disable_training_metrics_computation: bool = False,
        verbose: bool = True,
    ):
        """
        An object to configure the monitoring of the posterior fitting.

        Parameters
        ----------
        metrics: Optional[Callable[[jnp.ndarray], float]]
            Metrics to monitor during training.
        early_stopping_patience: int
            Number of consecutive epochs without an improvement in the performance on the validation set before stopping
             the training.
        early_stopping_monitor: str
            Validation metric to be monitored.
        early_stopping_min_delta: float
            Minimum change between updates to be considered an improvement, i.e., if the absolute change is less than
            `early_stopping_min_delta` then this is not considered an improvement leading to a potential early stop.
        eval_every_n_epochs: int
            Number of training epochs between validation set performance computation. To disable, set
            `eval_every_n_epochs` to None or 0 (i.e., no validation  metrics_names will be computed during training).
        disable_training_metrics_computation: bool
            if True, during training the only metric computed is the objective function.
            Otherwise, all the metrics_names provided by the user at runtime will be computed for the training step.
        verbose: bool
            Whether to log the training progress.
        """
        if metrics is not None:
            if type(metrics) != tuple:
                raise ValueError("`metrics` must be a tuple of callable metrics.")
            for metric in metrics:
                if not callable(metric):
                    raise ValueError(
                        f"All metrics in `metrics` must be callable objects, but {metric} is not."
                    )

        self.metrics = metrics
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_monitor = early_stopping_monitor
        self.early_stopping_min_delta = early_stopping_min_delta
        self.eval_every_n_epochs = eval_every_n_epochs
        self.disable_training_metrics_computation = disable_training_metrics_computation
        self.verbose = verbose
