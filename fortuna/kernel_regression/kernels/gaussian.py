import jax.numpy as jnp

from fortuna.typing import Array


def gaussian_kernel(x: Array) -> Array:
    return jnp.exp(-0.5 * x**2) / jnp.sqrt(2 * jnp.pi)
