from typing import (
    Any,
    Callable,
    Tuple,
)

import jax.numpy as jnp

from fortuna.typing import (
    Array,
    Batch,
    Params,
)


def scaled_mse_fn(
    apply: Callable[[Params, Array], jnp.ndarray],
    params: Params,
    batch: Batch,
) -> Tuple[jnp.ndarray, Any]:
    """
    Compute a variance-scaled mean-squared-error (MSE).

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
        Scaled MSE evalution and auxiliary objects.
    """
    outputs, aux = apply(params, batch[0])
    return scaled_mse_fn_from_outputs_and_targets(outputs, batch[1]), aux


def scaled_mse_fn_from_outputs_and_targets(
    outputs: Array,
    targets: Array,
) -> jnp.ndarray:
    means, log_vars = jnp.split(outputs, 2, axis=-1)
    return jnp.mean(jnp.sum(jnp.exp(-log_vars) * (targets - means) ** 2 + log_vars, -1))
