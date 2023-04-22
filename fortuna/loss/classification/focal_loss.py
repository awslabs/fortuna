from fortuna.typing import Array
import jax.numpy as jnp
from jax.nn import one_hot, softmax


def focal_loss_fn(
        outputs: Array,
        targets: Array,
        gamma: float = 2.,
) -> jnp.ndarray:
    probs = softmax(outputs, -1)
    targets = one_hot(targets, outputs.shape[-1])
    probs = jnp.sum(probs * targets, -1)
    return -jnp.mean((1 - probs) ** gamma * jnp.log(probs))
