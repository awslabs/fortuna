from typing import Callable, Optional

import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import Initializer

from fortuna.typing import Array


class ScalarConstantModel(nn.Module):
    r"""
    A scalar constant model, that is :math:`f(\theta, x) = \theta`, with :math:`\theta\in\mathbb{R}`. The scalar value
    will be broadcasted to the output dimension.

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
        scalar = self.param("scalar", self.initializer_fun, (1,))
        return jnp.broadcast_to(scalar, shape=(x.shape[0], self.output_dim))
