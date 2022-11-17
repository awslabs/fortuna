from typing import Any

import flax.linen as nn
import jax.numpy as jnp
from fortuna.typing import Array


class LeNet5(nn.Module):
    """
    A LeNet-5 network [LeCun et al., 1989](http://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf). Please refer to
    :class:`~fortuna.prob_model.model.base.Model` for the internal methods.

    Attributes
    ----------
    output_dim: int
        The output model dimension.
    dtype: Any
        Layers' dtype.
    """

    output_dim: int
    dtype: Any = jnp.float32

    def setup(self):
        self.dfe_subnet = LeNet5DeepFeatureExtractorSubNet(dtype=self.dtype)
        self.output_subnet = LeNet5OutputSubNet(
            output_dim=self.output_dim, dtype=self.dtype
        )

    def __call__(self, x: Array, **kwargs) -> jnp.ndarray:
        """
        Forward pass.

        Parameters
        ----------
        x: Array
            Inputs.

        jnp.ndarray
            Model outputs.
        """
        x = self.dfe_subnet(x)
        x = self.output_subnet(x)
        return x


class LeNet5DeepFeatureExtractorSubNet(nn.Module):
    """
    Deep feature extractor sub-network of a LeNet-5.

    Attributes
    ----------
    dtype: Any
        Layers' dtype.
    """

    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: Array):
        """
        Forward pass.

        Parameters
        ----------
        x: Array
            Inputs.

        jnp.ndarray
            Output of the hidden layers.
        """
        x = nn.Conv(
            features=6,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding="valid",
            dtype=self.dtype,
        )(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(
            features=16,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding="valid",
            dtype=self.dtype,
        )(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=120, dtype=self.dtype)(x)
        x = nn.relu(x)
        x = nn.Dense(features=84, dtype=self.dtype)(x)
        return x


class LeNet5OutputSubNet(nn.Module):
    """
    Output sub-network of a LeNet-5.

    Attributes
    ----------
    output_dim: int
        The output model dimension.
    dtype: Any
        Layers' dtype.
    """

    output_dim: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        """
        Forward pass.

        Parameters
        ----------
        x: jnp.ndarray
            Outputs of the deep feature extractor sub-network.

        Returns
        -------
        jnp.ndarray
            Model outputs.
        """
        x = nn.relu(x)
        x = nn.Dense(features=self.output_dim, dtype=self.dtype)(x)
        return x
