from typing import (
    Callable,
    Optional,
    Tuple,
    Union,
)

import jax.numpy as jnp

from fortuna.typing import Array


class Monitor:
    def __init__(
        self,
        metrics: Optional[
            Tuple[Callable[[jnp.ndarray, jnp.ndarray, Array], Union[float, Array]], ...]
        ] = None,
        uncertainty_fn: Optional[
            Callable[[jnp.ndarray, jnp.ndarray, Array], jnp.ndarray]
        ] = None,
        early_stopping_patience: int = 0,
        early_stopping_monitor: str = "val_loss",
        early_stopping_min_delta: float = 0.0,
        eval_every_n_epochs: int = 1,
        disable_calibration_metrics_computation: bool = False,
        verbose: bool = True,
    ):
        """
        An object to configure the monitoring of the calibration process.

        Parameters
        ----------
        metrics: Optional[Callable[[jnp.ndarray, jnp.ndarray, Array], Union[float, Array]]]
            Metrics to monitor during calibration. This must take three arguments: predictions, uncertainty estimates
            and target variables. In classification, :func:`.expected_calibration_error` is an example of valid metric.
        uncertainty_fn: Optional[Tuple[Callable[[jnp.ndarray, jnp.ndarray, Array], Union[float, Array]], ...]]
            A function that maps (calibrated) outputs into uncertainty estimates. These will be used in `metrics`.
            In classification,
            the default is :func:`~fortuna.prob_output_layer.classification.ProbOutputClassifier.mean`.
            In regression,
            the default is :func:`~fortuna.prob_output_layer.regression.ProbOutputRegressor.variance`.
        early_stopping_patience: int
            Number of consecutive epochs without an improvement in the performance on the validation set before stopping
             the calibration.
        early_stopping_monitor: str
            Validation metric to be monitored for early stopping.
        early_stopping_min_delta: float
            Minimum change between updates to be considered an improvement, i.e., if the absolute change is less than
            `early_stopping_min_delta` then this is not considered an improvement leading to a potential early stop.
        eval_every_n_epochs: int
            Number of calibration epochs between validation. To disable, set
            `eval_every_n_epochs` to None or 0 (i.e., no validation metrics will be computed during calibration).
        disable_calibration_metrics_computation: bool
            if True, during calibration the only metric computed is the objective function.
            Otherwise, all the metrics provided by the user at runtime will be computed for the training step.
        verbose: bool
            Whether to log the calibration progress.
        """
        if metrics is not None:
            if type(metrics) != tuple:
                raise ValueError("`metrics` must be a tuple of callable metrics.")
            for metric in metrics:
                if not callable(metric):
                    raise ValueError(
                        f"All metrics in `metrics` must be callable objects, but {metric} is not."
                    )
        if uncertainty_fn is not None and not callable(uncertainty_fn):
            raise ValueError("`uncertainty_fn` must be a a callable function.")

        self.metrics = metrics
        self.uncertainty_fn = uncertainty_fn
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_monitor = early_stopping_monitor
        self.early_stopping_min_delta = early_stopping_min_delta
        self.eval_every_n_epochs = eval_every_n_epochs
        self.disable_calibration_metrics_computation = (
            disable_calibration_metrics_computation
        )
        self.verbose = verbose
