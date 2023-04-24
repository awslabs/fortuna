"""
Wide ResNet model
(adapted from https://github.com/google/flax/blob/v0.2/examples/cifar10/models/wideresnet.py)
"""
from functools import partial
from typing import Any, Callable, Tuple

import flax.linen as nn
import jax.numpy as jnp

from fortuna.model.utils.spectral_norm import WithSpectralConv2DNormMixin
from fortuna.typing import Array

ModuleDef = Any


class WideResnetBlock(nn.Module):
    """
    A wide residual network block.

    Attributes
    ----------
    conv: ModuleDef
        Convolution module.
    norm: ModuleDef
        Normalization module.
    activation: Callable
        Activation function.
    filters: int
        Number of filters.
    strides: Tuple[int, int]
        Strides.
    dropout: ModuleDef
        Dropout module.
    dropout_rate: float
        Dropout rate.
    """

    conv: ModuleDef
    norm: ModuleDef
    activation: Callable
    filters: int
    strides: Tuple[int, int] = (1, 1)
    dropout: ModuleDef = nn.Dropout
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """
        Block forward pass.

        Parameters
        ----------
        x: jnp.ndarray
            Block inputs.
        train: bool
            Whether the call is performed during training.

        Returns
        -------
        jnp.ndarray
            Block outputs.
        """
        dropout = self.dropout(rate=self.dropout_rate, broadcast_dims=(1,2))

        y = self.norm(name="bn1")(x)
        y = nn.relu(y)
        if self.dropout_rate > 0.0:
            y = dropout(y, deterministic=not train)
        y = self.conv(self.filters, (3, 3), self.strides, name="conv1")(y)
        y = self.norm(name="bn2")(y)
        y = nn.relu(y)
        if self.dropout_rate > 0.0:
            y = dropout(y, deterministic=not train)
        y = self.conv(self.filters, (3, 3), name="conv2")(y)

        # Apply an up projection in case of channel mismatch
        if (x.shape[-1] != self.filters) or self.strides != (1, 1):
            x = self.conv(self.filters, (3, 3), self.strides)(x)
        return x + y


class WideResnetGroup(nn.Module):
    """
    A wide residual network group.

    Attributes
    ----------
    conv: ModuleDef
        Convolution module.
    norm: ModuleDef
        Normalization module.
    activation: Callable
        Activation function.
    blocks_per_group: int
        Number of blocks per group.
    strides: Tuple[int, int]
        Strides.
    dropout: ModuleDef
        Dropout module.
    dropout_rate: float
        Dropout rate.
    """

    conv: ModuleDef
    norm: ModuleDef
    activation: Callable
    blocks_per_group: int
    filters: int
    strides: Tuple[int, int] = (1, 1)
    dropout: ModuleDef = nn.Dropout
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """
        Group forward pass.

        Parameters
        ----------
        x: jnp.ndarray
            Group inputs.
        train: bool
            Whether the call is performed during training.

        Returns
        -------
        jnp.ndarray
            Group outputs.
        """
        for i in range(self.blocks_per_group):
            x = WideResnetBlock(
                conv=self.conv,
                norm=self.norm,
                activation=self.activation,
                filters=self.filters,
                strides=self.strides if i == 0 else (1, 1),
                dropout=self.dropout,
                dropout_rate=self.dropout_rate,
            )(x, train=train)
        return x


class DeepFeatureExtractorSubNet(nn.Module):
    """
    Deep feature extractor subnetwork.

    Attributes
    ----------
    depth: int
        Depth of the subnetwork.
    widen_factor: int
        Widening factor.
    dropout_rate: float
        Dropout rate.
    dtype: Any
        Layers' dtype.
    activation: Callable
        Activation function.
    conv: ModuleDef
        Convolution module.
    dropout: ModuleDef
        Dropout module.
    """

    depth: int = 28
    widen_factor: int = 10
    dropout_rate: float = 0.0
    dtype: Any = jnp.float32
    activation: Callable = nn.relu
    conv: ModuleDef = nn.Conv
    dropout: ModuleDef = nn.Dropout

    @nn.compact
    def __call__(self, x: Array, train: bool = True) -> jnp.ndarray:
        """
        Deep feature extractor subnetwork forward pass.

        Parameters
        ----------
        x: Array
            Input data.
        train: bool
            Whether the call is performed during training.

        Returns
        -------
        jnp.ndarray
            Deep feature extractor representation.
        """
        if hasattr(self, 'spectral_norm'):
            conv = self.spectral_norm(self.conv, train=train)
        else:
            conv = self.conv

        blocks_per_group = (self.depth - 4) // 6

        dropout = self.dropout(rate=self.dropout_rate, broadcast_dims=(1,2))
        conv = partial(conv, use_bias=False, dtype=self.dtype)
        norm = partial(
            nn.BatchNorm,
            use_running_average=not train,
            momentum=0.9,
            epsilon=1e-5,
            dtype=self.dtype,
        )

        x = conv(16, (3, 3), name="init_conv")(x)
        if self.dropout_rate > 0.0:
            x = dropout(x, deterministic=not train)
        x = WideResnetGroup(
            conv=conv,
            norm=norm,
            activation=self.activation,
            blocks_per_group=blocks_per_group,
            filters=16 * self.widen_factor,
            strides=(1, 1),
            dropout=self.dropout,
            dropout_rate=self.dropout_rate,
        )(x, train=train)
        x = WideResnetGroup(
            conv=conv,
            norm=norm,
            activation=self.activation,
            blocks_per_group=blocks_per_group,
            filters=32 * self.widen_factor,
            strides=(2, 2),
            dropout=self.dropout,
            dropout_rate=self.dropout_rate,
        )(x, train=train)
        x = WideResnetGroup(
            conv=conv,
            norm=norm,
            activation=self.activation,
            blocks_per_group=blocks_per_group,
            filters=64 * self.widen_factor,
            strides=(2, 2),
            dropout=self.dropout,
            dropout_rate=self.dropout_rate,
        )(x, train=train)
        x = norm()(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, (8, 8))
        x = x.reshape((x.shape[0], -1))
        return x


class OutputSubNet(nn.Module):
    """
    Output subnetwork.

    Parameters
    ----------
    output_dim: int
        Output dimension.
    dtype: Any
        Layers' dtype.
    """

    output_dim: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """
        Output subnetwork forward pass.

        Parameters
        ----------
        x: jnp.ndarray
            Subnetwork inputs.
        train: bool
            Whether the call is performed during training.

        Returns
        -------
        jnp.ndarray
            Outputs.
        """
        x = nn.Dense(self.output_dim, dtype=self.dtype)(x)
        return x


class WideResNet(nn.Module):
    """
    Wide residual network class.

    Attributes
    ----------
    output_dim: int
        Output dimension.
    depth: int
        Depth of the subnetwork.
    widen_factor: int
        Widening factor.
    dropout_rate: float
        Dropout rate.
    dtype: Any
        Layers' dtype.
    activation: Callable
        Activation function.
    conv: ModuleDef
        Convolution module.
    dropout: ModuleDef
        Dropout module.
    """

    output_dim: int
    depth: int = 28
    widen_factor: int = 10
    dropout_rate: float = 0.0
    dtype: Any = jnp.float32
    activation: Callable = nn.relu
    conv: ModuleDef = nn.Conv
    dropout: ModuleDef = nn.Dropout

    def setup(self):
        self.dfe_subnet = DeepFeatureExtractorSubNet(
            depth=self.depth,
            widen_factor=self.widen_factor,
            dropout_rate=self.dropout_rate,
            dtype=self.dtype,
            activation=self.activation,
            conv=self.conv,
            dropout=self.dropout,
        )
        self.output_subnet = OutputSubNet(output_dim=self.output_dim, dtype=self.dtype)

    def __call__(self, x: Array, train: bool = True) -> jnp.ndarray:
        """
        Forward pass.

        Parameters
        ----------
        x: Array
            Input data.
        train: bool
            Whether the call is performed during training.

        Returns
        -------
        jnp.ndarray
            Outputs.
        """
        x = self.dfe_subnet(x, train)
        x = self.output_subnet(x, train)
        return x


WideResNet28_10 = partial(WideResNet, depth=28, widen_factor=10)


class WideResNetDeepFeatureExtractorSubNetWithSN(WithSpectralConv2DNormMixin, DeepFeatureExtractorSubNet):
    pass

# define the feature extractors with spectral norm
WideResNetD28W10DeepFeatureExtractorSubNetWithSN = partial(WideResNetDeepFeatureExtractorSubNetWithSN, depth=28, widen_factor=10)