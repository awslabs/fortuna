import dataclasses
from functools import partial
from typing import Any, Optional, Tuple, Union

import jax.numpy as jnp

from fortuna.model.resnet import BottleneckResNetBlock
from fortuna.model.resnet import DeepFeatureExtractorSubNet as ResNetDeepFeatureExtractorSubNet
from fortuna.model.resnet import ResNet, ResNetBlock
from fortuna.model.utils.spectral_norm import WithSpectralConv2DNormMixin
from fortuna.model.wideresnet import DeepFeatureExtractorSubNet as WideResNetDeepFeatureExtractorSubNet
from fortuna.model.wideresnet import WideResNet
from fortuna.model.utils.random_features import RandomFeatureGaussianProcess
from fortuna.typing import Array

ModuleDef = Any


@dataclasses.dataclass
class SNGPMixin:
    spectral_norm_iteration: int = 1
    spectral_norm_bound: float = 0.9
    gp_hidden_features: int = 1024
    gp_input_dim: Optional[int] = None
    normalize_input: bool = False
    use_full_covmat: bool = True
    ridge_penalty: float = 1.0
    momentum: Optional[float] = None


class SNGPResNetDeepFeatureExtractorSubNet(WithSpectralConv2DNormMixin, ResNetDeepFeatureExtractorSubNet):
    pass


class ResNetSNGP(SNGPMixin, ResNet):
    """
    [Spectral-normalized Neural Gaussian Process](https://arxiv.org/abs/2006.10108) using a ResNet as the
    deep feature extractor.
    """
    def setup(self):
        self.dfe_subnet = SNGPResNetDeepFeatureExtractorSubNet(
            stage_sizes=self.stage_sizes,
            block_cls=self.block_cls,
            num_filters=self.num_filters,
            dtype=self.dtype,
            activation=self.activation,
            conv=self.conv,
            spectral_norm_bound=self.spectral_norm_bound,
            spectral_norm_iteration=self.spectral_norm_iteration,
        )
        self.output_subnet = RandomFeatureGaussianProcess(
            features=self.output_dim,
            hidden_features=self.gp_hidden_features,
            normalize_input=self.normalize_input,
            covmat_kwargs={
                "ridge_penalty": self.ridge_penalty,
                "momentum": self.momentum,
            },
        )

    def __call__(
        self, x: Array, train: bool = False, **kwargs
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        x = self.dfe_subnet(x, train)
        x = self.output_subnet(x, return_full_covmat=self.use_full_covmat)
        return x


class SNGPWideResNetDeepFeatureExtractorSubNet(WithSpectralConv2DNormMixin, WideResNetDeepFeatureExtractorSubNet):
    pass


class WideResNetSNGP(SNGPMixin, WideResNet):
    """
    [Spectral-normalized Neural Gaussian Process](https://arxiv.org/abs/2006.10108) using a WideResNet as the
    deep feature extractor.
    """
    def setup(self):
        self.dfe_subnet = SNGPWideResNetDeepFeatureExtractorSubNet(
            depth=self.depth,
            widen_factor=self.widen_factor,
            dropout_rate=self.dropout_rate,
            dtype=self.dtype,
            activation=self.activation,
            conv=self.conv,
            spectral_norm_bound=self.spectral_norm_bound,
            spectral_norm_iteration=self.spectral_norm_iteration,
        )
        self.output_subnet = RandomFeatureGaussianProcess(
            features=self.output_dim,
            hidden_features=self.gp_hidden_features,
            normalize_input=self.normalize_input,
            covmat_kwargs={
            "ridge_penalty": self.ridge_penalty,
            "momentum": self.momentum,
        },
        )

    def __call__(
        self, x: Array, train: bool = False, **kwargs
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        x = self.dfe_subnet(x, train)
        x = self.output_subnet(x, return_full_covmat=self.use_full_covmat)
        return x


ResNet18_SNGP = partial(ResNetSNGP, stage_sizes=[2, 2, 2, 2], block_cls=ResNetBlock)
ResNet34_SNGP = partial(ResNetSNGP, stage_sizes=[3, 4, 6, 3], block_cls=ResNetBlock)
ResNet50_SNGP = partial(ResNetSNGP, stage_sizes=[3, 4, 6, 3], block_cls=BottleneckResNetBlock)
ResNet101_SNGP = partial(ResNetSNGP, stage_sizes=[3, 4, 23, 3], block_cls=BottleneckResNetBlock)
ResNet152_SNGP = partial(ResNetSNGP, stage_sizes=[3, 8, 36, 3], block_cls=BottleneckResNetBlock)
ResNet200_SNGP = partial(ResNetSNGP, stage_sizes=[3, 24, 36, 3], block_cls=BottleneckResNetBlock)
WideResNet28_10_SNGP = partial(WideResNetSNGP, depth=28, widen_factor=10)