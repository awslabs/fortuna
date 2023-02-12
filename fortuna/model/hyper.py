import flax.linen as nn
import jax.numpy as jnp

from fortuna.typing import Array


class HyperparameterModel(nn.Module):
    r"""
    A hyperparameter model. The value of the hyperparameter will not change during training.

    Parameters
    ----------
    value: Union[float, Array]
        Value of the hyperparameter.
    """
    value: Array

    def setup(self) -> None:
        if self.value.ndim != 1:
            raise ValueError(
                "`value` must be a one-dimensional array, with length equal to the output dimension of "
                "the model."
            )
        dummy = self.param("none", nn.initializers.zeros, (0,))

    def __call__(self, x: Array, **kwargs) -> jnp.ndarray:
        return jnp.broadcast_to(self.value, shape=(x.shape[0], len(self.value)))
