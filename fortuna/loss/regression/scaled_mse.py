import jax.numpy as jnp

from fortuna.typing import Array


def scaled_mse_fn(
    outputs: Array,
    targets: Array,
) -> jnp.ndarray:
    """
    Compute a variance-scaled mean-squared-error (MSE).

    Parameters
    ----------
    outputs: Array
        Model outputs to be passed to the loss.
    targets: Array
        Target data points.

    Returns
    -------
    Tuple[jnp.ndarray, Any]
        Scaled MSE evalution and auxiliary objects.
    """
    means, log_vars = jnp.split(outputs, 2, axis=-1)
    return jnp.mean(jnp.sum(jnp.exp(-log_vars) * (targets - means) ** 2 + log_vars, -1))
