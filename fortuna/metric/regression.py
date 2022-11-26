import jax.numpy as jnp

from fortuna.typing import Array


def root_mean_squared_error(preds: Array, targets: Array) -> float:
    """
    Compute the root-mean-squared error (RMSE).

    Parameters
    ----------
    preds: Array
        A two-dimensional array of predictions over the data points.
    targets: Array
        A two-dimensional array of target variables.

    Returns
    -------
    float
        The computed RMSE.
    """
    return jnp.sqrt(jnp.mean(jnp.sum((preds - targets) ** 2, axis=1)))


def rmse(preds: Array, targets: Array) -> float:
    """See :func:`.root_mean_squared_error`."""
    return root_mean_squared_error(preds, targets)


def mean_squared_error(preds: Array, targets: Array) -> float:
    """
    Compute the mean-squared error (MSE).

    Parameters
    ----------
    preds: Array
        A two-dimensional array of predictions over the data points.
    targets: Array
        A two-dimensional array of target variables.

    Returns
    -------
    float
        The computed MSE.
    """
    return jnp.mean(jnp.sum((preds - targets) ** 2, axis=1))


def mse(preds: Array, targets: Array) -> float:
    """See :func:`.mean_squared_error`."""
    return mean_squared_error(preds, targets)


def root_mean_absolute_error(preds: Array, targets: Array) -> float:
    """
    Compute the root-mean-absolute error (RMAE).

    Parameters
    ----------
    preds: Array
        A two-dimensional array of predictions over the data points.
    targets: Array
        A two-dimensional array of target variables.

    Returns
    -------
    float
        The computed RMAE.
    """
    return jnp.sqrt(jnp.mean(jnp.sum(jnp.abs(preds - targets), axis=1)))


def rmae(preds: Array, targets: Array) -> float:
    """See :func:`.root_mean_absolute_error`."""
    return root_mean_absolute_error(preds, targets)


def mean_absolute_error(preds: Array, targets: Array) -> float:
    """
    Compute the mean-absolute error (MAE).

    Parameters
    ----------
    preds: Array
        A two-dimensional array of predictions over the data points.
    targets: Array
        A two-dimensional array of target variables.

    Returns
    -------
    float
        The computed MAE.
    """
    return jnp.mean(jnp.sum(jnp.abs(preds - targets), axis=1))


def mae(preds: Array, targets: Array) -> float:
    """See :func:`.mean_absolute_error`."""
    return mean_absolute_error(preds, targets)


def prediction_interval_coverage_probability(
    lower_bounds: Array, upper_bounds: Array, targets: Array
) -> float:
    """
    Compute the prediction interval coverage probability (PICP). This is the fraction of data points for which the
    true targets lie within the estimated interval. This is supported only for scalar target data.

    Parameters
    ----------
    lower_bounds: Array
        Predictive lower bounds. These are the lower bounds of the estimated predictive intervals. This can either
        be a one-dimensional array with entry corresponding to different data points, or a two-dimensional array
        with first axis corresponding to different data points, and second axis with only one dimension.
    upper_bounds: Array
        Predictive upper bounds. These are the upper bounds of the estimated predictive intervals. This can either
        be a one-dimensional array with entry corresponding to different data points, or a two-dimensional array
        with first axis corresponding to different data points, and second axis with only one dimension.
    targets: Array
        A two-dimensional array of target variables.

    Returns
    -------
    float
        The computed PICP.
    """
    if targets.shape[1] > 1:
        raise ValueError(
            """This metric is supported only for target data such that `target.shape[1] == 1`, but 
        `target.shape[1] == {}` was found.""".format(
                targets.shape[1]
            )
        )
    if upper_bounds.ndim == 1:
        upper_bounds = upper_bounds[:, None]
    if upper_bounds.shape[1] != 1:
        raise ValueError(
            """The second axis of `upper_bounds` must contain only one component."""
        )
    if lower_bounds.ndim == 1:
        lower_bounds = lower_bounds[:, None]
    if lower_bounds.shape[1] != 1:
        raise ValueError(
            """The second axis of `lower_bounds` must contain only one component."""
        )

    satisfies_upper = targets <= upper_bounds
    satisfies_lower = targets >= lower_bounds
    return jnp.mean(satisfies_lower * satisfies_upper)


def picp(lower_bounds: Array, upper_bounds: Array, targets: Array) -> float:
    """See :func:`.prediction_interval_coverage_probability`."""
    return prediction_interval_coverage_probability(lower_bounds, upper_bounds, targets)
