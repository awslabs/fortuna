from typing import Optional

import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import Initializer

from fortuna.typing import Array


class ConstantModel(nn.Module):
    r"""
    A constant model, that is :math:`f(\theta, x) = \theta`.

    Parameters
    ----------
    output_dim: int
        The output model dimension.
    initializer_fun: Optional[Initializer]
        Function to initialize the model parameters.
        This must be one of the available options in :code:`flax.linen.initializers`.
    """
    output_dim: int
    initializer_fun: Optional[Initializer] = nn.initializers.zeros

    @nn.compact
    def __call__(self, x: Array, **kwargs) -> jnp.ndarray:
        constant = self.param("constant", self.initializer_fun, (self.output_dim,))
        return jnp.broadcast_to(constant, shape=(x.shape[0], self.output_dim))
