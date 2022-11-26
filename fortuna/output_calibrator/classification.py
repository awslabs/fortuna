import flax.linen as nn
import jax.numpy as jnp

from fortuna.typing import Array


class ClassificationTemperatureScaler(nn.Module):
    r"""
    Classification temperature scaling. It scales the logits with a scalar temperature parameters. Let :math:`o` be
    output logits and :math:`\phi` be a scalar parameter. Then the scaling can be seen as
    :math:`g(\phi, o) = \exp(-\phi) o`.
    """

    @nn.compact
    def __call__(self, x: Array, **kwargs) -> jnp.ndarray:
        log_temp = self.param("log_temp", nn.initializers.zeros, (1,))
        return x * jnp.exp(-log_temp)
