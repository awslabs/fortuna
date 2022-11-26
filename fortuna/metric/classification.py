from typing import Tuple, Union, Optional, Dict

import jax.nn
import jax.numpy as jnp

from fortuna.data.loader import TargetsLoader
from fortuna.plot import plot_reliability_diagram
from fortuna.typing import Array


def accuracy(preds: Array, targets: Array) -> jnp.ndarray:
    """
    Compute the accuracy given predictions and target variables.

    Parameters
    ----------
    preds: Array
        A one-dimensional array of predictions over the data points.
    targets: Array
        A one-dimensional array of target variables.

    Returns
    -------
    float
        The computed accuracy.
    """
    if preds.ndim > 1:
        raise ValueError(
            """`preds` must be a one-dimensional array of predicted classes."""
        )
    if targets.ndim > 1:
        raise ValueError(
            """`targets` must be a one-dimensional array of target classes."""
        )
    return jnp.mean(preds == targets)


def compute_counts_confs_accs(
    preds: Array, probs: Array, targets: Array, plot: bool = False, plot_options: Optional[Dict] = None
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Bin the confidence scores (maximum probability) and for each of them compute:
        - the number of inputs;
        - the average confidence score for each bin;
        - the average accuracy over each bin.

    Parameters
    ----------
    preds: Array
        A one-dimensional array of predictions over the data points.
    probs: Array
        A two-dimensional array of class probabilities for each data point.
    targets: Array
        A one-dimensional array of target variables.
    plot: bool
        Whether to plot a reliability diagram.
    plot_options: dict
        Options for the reliability diagram plot; see :func:`~fortuna.plot.plot_reliability_diagram`.

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        Number of inputs per bin, average confidence score per bin and average accuracy per bin.
    """
    if probs.ndim != 2:
        raise ValueError("""`probs` must be a two-dimensional array.""")
    thresholds = jnp.linspace(1 / probs.shape[1], 1, 10)
    probs = probs.max(1)
    probs = jnp.array(probs)
    indices = [jnp.where(probs <= thresholds[0])[0]]
    indices += [
        jnp.where((probs <= thresholds[i]) & (probs > thresholds[i - 1]))[0]
        for i in range(1, len(thresholds))
    ]
    counts = jnp.array([len(idx) for idx in indices])

    diff = targets - preds
    accs = jnp.array([jnp.nan_to_num(jnp.mean(diff[idx] == 0)) for idx in indices])
    confs = jnp.array([jnp.nan_to_num(jnp.mean(probs[idx])) for idx in indices])

    if plot:
        idx = confs != 0
        plot_reliability_diagram(accs[idx], confs[idx], **plot_options)

    return counts, confs, accs


def expected_calibration_error(
    preds: Array, probs: Array, targets: Array, plot: bool = False, plot_options: Optional[Dict] = None
) -> jnp.ndarray:
    """
    Compute the Expected Calibration Error (ECE)
    (see `Naeini et al., 2015 <https://people.cs.pitt.edu/~milos/research/AAAI_Calibration.pdf>`__ and
    `Guo et al., 2017 <http://proceedings.mlr.press/v70/guo17a/guo17a.pdf>`__). Optionally, plot and save a reliability
    diagram.

    Parameters
    ----------
    preds: Array
        A one-dimensional array of predictions over the data points.
    probs: Array
        A two-dimensional array of class probabilities for each data point.
    targets: Array
        A one-dimensional array of target variables.
    plot: bool
        Whether to plot a reliability diagram.
    plot_options: dict
        Options for the reliability diagram plot; see :func:`~fortuna.plot.plot_reliability_diagram`.

    Returns
    -------
    float
        The value of the ECE.
    """
    counts, confs, accs = compute_counts_confs_accs(
        preds, probs, targets, plot, plot_options
    )
    ece = jnp.sum(counts * (accs - confs) ** 2) / preds.shape[0]
    return ece


def ece(
    preds: Array, probs: Array, targets: Array, plot: bool = False, plot_options: Optional[Dict] = None
) -> float:
    """See :func:`.expected_calibration_error`."""
    return expected_calibration_error(preds, probs, targets, plot, plot_options)


def maximum_calibration_error(
    preds: Array, probs: Array, targets: Array, plot: bool = False, plot_options: Optional[Dict] = None
) -> jnp.ndarray:
    """
    Compute the Maximum Calibration Error (MCE)
    (see `Naeini et al., 2015 <https://people.cs.pitt.edu/~milos/research/AAAI_Calibration.pdf>`__). Optionally, plot
    and save a reliability diagram.

    Parameters
    ----------
    preds: Array
        A one-dimensional array of predictions over the data points.
    probs: Array
        A two-dimensional array of class probabilities for each data point.
    targets: Array
        A one-dimensional array of target variables.
    plot: bool
        Whether to plot a reliability diagram.
    plot_options: dict
        Options for the reliability diagram plot; see :func:`~fortuna.plot.plot_reliability_diagram`.

    Returns
    -------
    float
        The value of the MCE.
    """
    counts, confs, accs = compute_counts_confs_accs(
        preds, probs, targets, plot, plot_options
    )
    mce = jnp.max(counts * (accs - confs) ** 2)
    return mce


def mce(
    preds: Array, probs: Array, targets: Array, plot: bool = False, plot_options: Optional[Dict] = None
) -> float:
    """See :func:`.maximum_calibration_error`."""
    return maximum_calibration_error(preds, probs, targets, plot, plot_options)


def brier_score(probs: Array, targets: Union[TargetsLoader, Array]) -> jnp.ndarray:
    """
    Brier score (see `Brier, 1950 <https://web.archive.org/web/20171023012737/
    https://docs.lib.noaa.gov/rescue/mwr/078/mwr-078-01-0001.pdf>`__). This can be used for both binary and multi-class
    classification.

    Parameters
    ----------
    probs: Array
        A two-dimensional array of class probabilities for each data point.
    targets: Array
        A one-dimensional array of target variables.

    Returns
    -------
    float
        The Brier score.
    """
    if probs.ndim != 2:
        raise ValueError(
            """`probs` must be a two-dimensional array of probabilities for each class and each data
        point."""
        )
    if type(targets) == TargetsLoader:
        targets = targets.to_array_targets()
    targets = jax.nn.one_hot(targets, probs.shape[1])
    return jnp.mean(jnp.sum((probs - targets) ** 2, axis=1))
