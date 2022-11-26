import flax.linen as nn
import jax.numpy as jnp

from fortuna.typing import Array


class RegressionTemperatureScaler(nn.Module):
    r"""
    Regression temperature scaling. It multiplies the variance with a scalar temperature parameters. Let :math:`v` be
    the variance outputs and :math:`\phi` be a scalar parameter. Then the scaling can be seen as
    :math:`g(\phi, o) = \exp(\phi) v`.
    """

    @nn.compact
    def __call__(self, x: Array, **kwargs) -> jnp.ndarray:
        log_temp = self.param("log_temp", nn.initializers.zeros, (1,))
        mean, log_var = jnp.split(x, 2, axis=-1)
        log_var += log_temp
        return jnp.concatenate((mean, log_var), axis=-1)
