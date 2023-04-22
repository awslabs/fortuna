from fortuna.typing import Array
import jax.numpy as jnp
import jax.scipy as jsp
import jax


def cross_entropy_loss_fn(
        outputs: Array,
        targets: Array,
) -> jnp.ndarray:
    targets = jax.nn.one_hot(targets, outputs.shape[-1])
    return jnp.mean(jsp.special.logsumexp(outputs, -1) - jnp.sum(targets * outputs, -1))
