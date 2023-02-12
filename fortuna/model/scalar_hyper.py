import flax.linen as nn
import jax.numpy as jnp

from fortuna.typing import Array


class ScalarHyperparameterModel(nn.Module):
    r"""
    A scalar hyperparameter model. The scalar value of the hyperparameter will not change during training, and it will
    be broadcasted to the output dimension.

    Parameters
    ----------
    output_dim: int
        The output model dimension.
    value: float
        Scalar value of the hyperparameter.
    """
    output_dim: int
    value: float

    def setup(self) -> None:
        if type(self.value) != float:
            raise ValueError(
                f"`value` must be a float, but a {type(self.value)} was found instead."
            )
        dummy = self.param("none", nn.initializers.zeros, (0,))

    def __call__(self, x: Array, **kwargs) -> jnp.ndarray:
        return jnp.broadcast_to(self.value, shape=(x.shape[0], self.output_dim))
