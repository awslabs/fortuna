from fortuna.typing import Array
import jax.numpy as jnp
from jax.nn import one_hot, softmax


def focal_loss_fn(
        outputs: Array,
        targets: Array,
        gamma: float = 2.,
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
