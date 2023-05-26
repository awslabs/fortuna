from typing import (
    Any,
    Callable,
    Tuple,
)

import jax
from jax import value_and_grad
import jax.numpy as jnp
import jax.scipy as jsp

from fortuna.typing import (
    Array,
    Batch,
    Params,
)


def cross_entropy_loss_fn(
    apply: Callable[[Params, Array], jnp.ndarray],
    params: Params,
    batch: Batch,
) -> Tuple[jnp.ndarray, Any]:
    """
    A cross-entropy loss function. Check `here <https://en.wikipedia.org/wiki/Cross_entropy>`_ for reference.

    Parameters
    ----------
    apply: Callable[[Params, Batch], jnp.ndarray]
        A function evaluating the model.
    params: Params
        Model parameters.
    batch: Batch
        Batch of data points.

    Returns
    -------
    Tuple[jnp.ndarray, Any]
        The cross-entropy loss evaluation and auxiliary objects.
    """
    outputs, aux = apply(params, batch[0])
    return cross_entropy_loss_fn_from_outputs_and_targets(outputs, batch[1]), aux


def cross_entropy_loss_fn_from_outputs_and_targets(
    outputs: jnp.ndarray,
    targets: Array,
) -> jnp.ndarray:
    targets = jax.nn.one_hot(targets, outputs.shape[-1])
    return jnp.mean(jsp.special.logsumexp(outputs, -1) - jnp.sum(targets * outputs, -1))
