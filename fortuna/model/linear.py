import flax.linen as nn
import jax.numpy as jnp

from fortuna.typing import Array


class Linear(nn.Module):
    """
    A linear model.

    Parameters
    ----------
    output_dim: int
        The output model dimension.
    """

    output_dim: int

    @nn.compact
    def __call__(self, x: Array, **kwargs) -> jnp.ndarray:
        x = nn.Dense(self.output_dim, name="last")(x)
        return x
