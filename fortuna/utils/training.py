import jax.numpy as jnp
from jax.tree_util import tree_map, tree_reduce


def clip_grandients_by_value(grad: jnp.ndarray, max_grad_val: float) -> jnp.ndarray:
    clip_fn = lambda z: jnp.clip(z, -max_grad_val, max_grad_val)
    grad = tree_map(clip_fn, grad)
    return grad


def clip_grandients_by_norm(grad: jnp.ndarray, max_grad_norm: float) -> jnp.ndarray:
    grad_norm = jnp.sqrt(
        tree_reduce(lambda x, y: x + jnp.sum(y**2), grad, initializer=0)
    )
    mult = jnp.minimum(1, max_grad_norm / (1e-7 + grad_norm))
    grad = tree_map(lambda z: mult * z, grad)
    return grad
