"""
Flax implementation of ResNet V1.
Taken as is from https://github.com/google/flax/blob/main/examples/imagenet/models.py
"""

# See issue #620.
# pytype: disable=wrong-arg-count

from functools import partial
from typing import Any, Callable, Sequence, Tuple

import flax.linen as nn
import jax.numpy as jnp

from fortuna.model.utils.spectral_norm import WithSpectralConv2DNorm
from fortuna.typing import Array

ModuleDef = Any


class ResNetBlock(nn.Module):
    """
    Residual network block.

    Attributes
    ----------
    filters: int
        Number of filters.
    conv: ModuleDef
        Convolution module.
    norm: ModuleDef
        Normalization module.
    activation: Callable
        Activation function.
    strides: Tuple[int, int]
        Strides.
    """

    filters: int
    conv: ModuleDef
    norm: ModuleDef
    activation: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Block forward pass.

        Parameters
        ----------
        x: jnp.ndarray
            Block inputs.

        Returns
        -------
        jnp.ndarray
            Block outputs.
        """
        residual = x
        y = self.conv(self.filters, (3, 3), self.strides)(x)
        y = self.norm()(y)
        y = self.activation(y)
        y = self.conv(self.filters, (3, 3))(y)
        y = self.norm(scale_init=nn.initializers.zeros)(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters, (1, 1), self.strides, name="conv_proj")(
                residual
            )
            residual = self.norm(name="norm_proj")(residual)

        return self.activation(residual + y)


class BottleneckResNetBlock(nn.Module):
    """
    Bottleneck residual network block.

    Attributes
    ----------
    filters: int
        Number of filters.
    conv: ModuleDef
        Convolution module.
    norm: ModuleDef
        Normalization module.
    activation: Callable
        Activation function.
    strides: Tuple[int, int]
        Strides.
    """

    filters: int
    conv: ModuleDef
    norm: ModuleDef
    activation: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Bottleneck block forward pass.

        Parameters
        ----------
        x: jnp.ndarray
            Block inputs.

        Returns
        -------
        jnp.ndarray
            Block outputs.
        """
        residual = x
        y = self.conv(self.filters, (1, 1))(x)
        y = self.norm()(y)
        y = self.activation(y)
        y = self.conv(self.filters, (3, 3), self.strides)(y)
        y = self.norm()(y)
        y = self.activation(y)
        y = self.conv(self.filters * 4, (1, 1))(y)
        y = self.norm(scale_init=nn.initializers.zeros)(y)

        if residual.shape != y.shape:
            residual = self.conv(
                self.filters * 4, (1, 1), self.strides, name="conv_proj"
            )(residual)
            residual = self.norm(name="norm_proj")(residual)

        return self.activation(residual + y)


class DeepFeatureExtractorSubNet(nn.Module):
    """
     Deep feature extractor subnetwork.

    Attributes
    ----------
    stage_sizes: Sequence[int]
        Sizes for each stage.
    block_cls: ModuleDef
        Block class.
    num_filters: int
        Number of filters.
    dtype: Any
        Layers' dtype.
    activation: Callable
        Activation function.
    conv: ModuleDef
        Convolution module.
    """

    stage_sizes: Sequence[int]
    block_cls: ModuleDef
    num_filters: int = 64
    dtype: Any = jnp.float32
    activation: Callable = nn.relu
    conv: ModuleDef = nn.Conv

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
        conv = partial(conv, use_bias=False, dtype=self.dtype)
        norm = partial(
            nn.BatchNorm,
            use_running_average=not train,
            momentum=0.9,
            epsilon=1e-5,
            dtype=self.dtype,
        )
        x = conv(
            self.num_filters, (7, 7), (2, 2), padding=[(3, 3), (3, 3)], name="conv_init"
        )(x)
        x = norm(name="bn_init")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")
        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = self.block_cls(
                    self.num_filters * 2**i,
                    strides=strides,
                    conv=conv,
                    norm=norm,
                    activation=self.activation,
                )(x)
        x = jnp.mean(x, axis=(1, 2))
        return x


class OutputSubNet(nn.Module):
    """
    Output subnetwork.

    Attributes
    ----------
    output_dim: int
        Output dimension.
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
            Deep feature extractor representation.
        train: bool
            Whether the call is performed during training.

        Returns
        -------
        jnp.ndarray
            Output of the subnetwork.
        """
        x = nn.Dense(self.output_dim, dtype=self.dtype)(x)
        x = jnp.asarray(x, self.dtype)
        return x


class ResNet(nn.Module):
    """
     Deep feature extractor subnetwork.

    Attributes
    ----------
    stage_sizes: Sequence[int]
        Sizes for each stage.
    block_cls: ModuleDef
        Block class.
    output_dim: int
        Output dimension.
    num_filters: int
        Number of filters.
    dtype: Any
        Layers' dtype.
    activation: Callable
        Activation function.
    conv: ModuleDef
        Convolution module.
    """

    stage_sizes: Sequence[int]
    block_cls: ModuleDef
    output_dim: int
    num_filters: int = 64
    dtype: Any = jnp.float32
    activation: Callable = nn.relu
    conv: ModuleDef = nn.Conv

    def setup(self):
        self.dfe_subnet = DeepFeatureExtractorSubNet(
            stage_sizes=self.stage_sizes,
            block_cls=self.block_cls,
            num_filters=self.num_filters,
            dtype=self.dtype,
            activation=self.activation,
            conv=self.conv,
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


ResNet18 = partial(ResNet, stage_sizes=[2, 2, 2, 2], block_cls=ResNetBlock)
ResNet34 = partial(ResNet, stage_sizes=[3, 4, 6, 3], block_cls=ResNetBlock)
ResNet50 = partial(ResNet, stage_sizes=[3, 4, 6, 3], block_cls=BottleneckResNetBlock)
ResNet101 = partial(ResNet, stage_sizes=[3, 4, 23, 3], block_cls=BottleneckResNetBlock)
ResNet152 = partial(ResNet, stage_sizes=[3, 8, 36, 3], block_cls=BottleneckResNetBlock)
ResNet200 = partial(ResNet, stage_sizes=[3, 24, 36, 3], block_cls=BottleneckResNetBlock)


class ResNetDeepFeatureExtractorSubNetWithSN(WithSpectralConv2DNorm, DeepFeatureExtractorSubNet):
    pass

# define the feature extractors with spectral norm
ResNet18DeepFeatureExtractorSubNetWithSN = partial(ResNetDeepFeatureExtractorSubNetWithSN, stage_sizes=[2, 2, 2, 2], block_cls=ResNetBlock)
ResNet34DeepFeatureExtractorSubNetWithSN = partial(ResNetDeepFeatureExtractorSubNetWithSN, stage_sizes=[3, 4, 6, 3], block_cls=ResNetBlock)
ResNet50DeepFeatureExtractorSubNetWithSN = partial(ResNetDeepFeatureExtractorSubNetWithSN, stage_sizes=[3, 4, 6, 3], block_cls=BottleneckResNetBlock)
ResNet101DeepFeatureExtractorSubNetWithSN = partial(ResNetDeepFeatureExtractorSubNetWithSN, stage_sizes=[3, 4, 23, 3], block_cls=BottleneckResNetBlock)
ResNet152DeepFeatureExtractorSubNetWithSN = partial(ResNetDeepFeatureExtractorSubNetWithSN, stage_sizes=[3, 8, 36, 3], block_cls=BottleneckResNetBlock)
ResNet200DeepFeatureExtractorSubNetWithSN = partial(ResNetDeepFeatureExtractorSubNetWithSN, stage_sizes=[3, 24, 36, 3], block_cls=BottleneckResNetBlock)
