from typing import Any

import flax.linen as nn
import jax.numpy as jnp


class CNN(nn.Module):
    output_dim: int
    dropout_rate: float
    dtype: Any = jnp.float32
    """
    A CNN model.

    :param output_dim: int
        Output dimension.
    :param dropout_rate: Optional[float]
        Dropout rate.
    :param dtype: Any
        Data type. Default: `float32`.
    """

    def setup(self):
        self.hidden_layers = CNNHiddenLayers(
            dropout_rate=self.dropout_rate, dtype=self.dtype
        )
        self.last_layer = CNNLastLayer(output_dim=self.output_dim, dtype=self.dtype)

    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        x = self.hidden_layers(x, train)
        x = self.last_layer(x, train)
        return x


class CNNHiddenLayers(nn.Module):
    dropout_rate: float
    dtype: Any = jnp.float32
    """
    Hidden layers of a CNN model.

    :param dropout_rate: float
        Dropout rate.
    :param dtype: Any
        Data type. Default: `float32`.
    """

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True):
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="valid",
            dtype=self.dtype,
        )(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Dropout(self.dropout_rate)(x, deterministic=not train)
        x = x.reshape((x.shape[0], -1))
        return x


class CNNLastLayer(nn.Module):
    output_dim: int
    dtype: Any = jnp.float32
    """
    Last layer of a CNN model.

    :param output_dim: int
        Output dimension.
    :param dtype: Any
        Data type. Default: `float32`.
    """

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True):
        x = nn.Dense(features=128, dtype=self.dtype)(x)
        x = nn.Dense(features=self.output_dim, dtype=self.dtype)(x)
        return x
