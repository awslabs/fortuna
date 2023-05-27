from typing import (
    Any,
    Callable,
    Tuple,
)

from jax.nn import (
    one_hot,
    softmax,
)
import jax.numpy as jnp

from fortuna.typing import (
    Array,
    Batch,
    Params,
)


def focal_loss_fn(
    apply: Callable[[Params, Array], jnp.ndarray],
    params: Params,
    batch: Batch,
    gamma: float = 2.0,
) -> Tuple[jnp.ndarray, Any]:
    """
    A focal loss function. See `[Mukhoti J. et a., 2020] <https://proceedings.neurips.cc/paper/2020/file/aeb7b30ef1d024a76f21a1d40e30c302-Paper.pdf>`_
    for reference.

    Parameters
    ----------
    apply: Callable[[Params, Batch], jnp.ndarray]
        A function evaluating the model.
    params: Params
        Model parameters.
    batch: Batch
        Batch of data points.
    gamma: float
        Hyper-parameter of the focal loss.

    Returns
    -------
    Tuple[jnp.ndarray, Any]
        The focal loss evaluation and auxiliary objects.
    """
    outputs, aux = apply(params, batch[0])
    return focal_loss_fn_from_outputs_and_targets(outputs, batch[1], gamma), aux


def focal_loss_fn_from_outputs_and_targets(
    outputs: Array,
    targets: Array,
    gamma: float = 2.0,
) -> jnp.ndarray:
    """
    A focal loss function. See `[Mukhoti J. et a., 2020] <https://proceedings.neurips.cc/paper/2020/file/aeb7b30ef1d024a76f21a1d40e30c302-Paper.pdf>`_
    for reference.

    Parameters
    ----------
    outputs: Array
        Model outputs to be passed to the loss.
    targets: Array
        Target data points.
    gamma: float
        Hyper-parameter of the focal loss.

    Returns
    -------
    jnp.ndarray
        The focal loss evaluation.
    """
    probs = softmax(outputs, -1)
    targets = one_hot(targets, outputs.shape[-1])
    probs = jnp.sum(probs * targets, -1)
    return -jnp.mean((1 - probs) ** gamma * jnp.log(probs))
