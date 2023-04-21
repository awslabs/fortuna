from fortuna.typing import Array
import jax.numpy as jnp
from jax.nn import one_hot
from jax.scipy.special import logsumexp


def cross_entropy_loss_fn(
        outputs: Array,
        targets: Array,
) -> jnp.ndarray:
    targets = one_hot(targets, outputs.shape[-1])
    return logsumexp(outputs, -1) - jnp.sum(targets * outputs, -1)
