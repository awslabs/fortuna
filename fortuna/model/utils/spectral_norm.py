"""
The code has been taken from https://github.com/google/edward2/blob/main/edward2/jax/nn/normalization.py
"""
import dataclasses
from typing import Any, Callable, Mapping, Optional, Tuple, Type

import flax.core
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax.random import PRNGKeyArray

from fortuna.typing import Array, Shape


def _l2_normalize(x: Array, eps: float = 1e-12) -> Array:
    return x * jax.lax.rsqrt(jnp.maximum(jnp.square(x).sum(), eps))


class SpectralNormalization(nn.Module):
    """
    Implements spectral normalization for linear layers.

    See `Spectral Normalization for Generative Adversarial Networks <https://arxiv.org/abs/1802.05957>`_ .

    Attributes
    ----------
    layer: nn.Module
        A Flax layer to apply normalization to.
    iteration: int
        The number of power iterations to estimate weight matrix's singular value.
    norm_multiplier: float
        Multiplicative constant to threshold the normalization.
        Usually under normalization, the singular value will converge to this value.
    u_init: Callable[[PRNGKeyArray, Shape, Type], Array]
        Initializer function for the first left singular vectors of the kernel.
    v_init: Callable[[PRNGKeyArray, Shape, Type], Array]
        Initializer function for the first right singular vectors of the kernel.
    kernel_apply_kwargs: Optional[Mapping[str, Any]]
        Updated keyword arguments to clone the input layer.
        The cloned layer represents the linear operator performed by the weight matrix.
        If not specified, that operator follows SN-GAN implementation
        (`Takeru M. et al <https://arxiv.org/abs/1802.05957>`_).
        In particular, for Dense layers the default behavior is equivalent to using a cloned layer with no bias
        (by specifying `kernel_apply_kwargs=dict(use_bias=False)`). With this customization, we
        can have the same implementation (inspried by
        (`Stephan H., 2020 <https://nbviewer.jupyter.org/gist/shoyer/fa9a29fd0880e2e033d7696585978bfc>`_)
        for different interpretations of Conv layers. Also see `SpectralNormalizationConv2D` for
        an example of using this attribute.
    kernel_name: str
        Name of the kernel parameter of the input layer.
    layer_name: Optional[str]
        Name of the input layer
    update_singular_value_estimate: Optional[bool]
        Whether to perform power interations to update the singular value estimate.
    """

    layer: nn.Module
    iteration: int = 1
    norm_multiplier: float = 0.95
    u_init: Callable[[PRNGKeyArray, Shape, Type], Array] = nn.initializers.normal(
        stddev=0.05
    )
    v_init: Callable[[PRNGKeyArray, Shape, Type], Array] = nn.initializers.normal(
        stddev=0.05
    )
    kernel_apply_kwargs: Optional[Mapping[str, Any]] = None
    kernel_name: str = "kernel"
    layer_name: Optional[str] = None
    update_singular_value_estimate: Optional[bool] = None

    def _get_singular_vectors(
        self, initializing: bool, kernel_apply: Callable, in_shape: Shape, dtype: Type
    ) -> Tuple[nn.Variable, nn.Variable]:
        if initializing:
            rng_u = self.make_rng("params")
            rng_v = self.make_rng("params")
            # Interpret output shape (not that this does not cost any FLOPs).
            out_shape = jax.eval_shape(
                kernel_apply, jax.ShapeDtypeStruct(in_shape, dtype)
            ).shape
        else:
            rng_u = rng_v = out_shape = None
        u = self.variable("spectral_stats", "u", self.u_init, rng_u, out_shape, dtype)
        v = self.variable("spectral_stats", "v", self.v_init, rng_v, in_shape, dtype)
        return u, v

    @nn.compact
    def __call__(
        self, inputs: Array, update_singular_value_estimate: Optional[bool] = None
    ) -> Array:
        """
        Applies a linear transformation with spectral normalization to the inputs.

        Parameters
        ----------
        inputs: Array
            The nd-array to be transformed.
        update_singular_value_estimate: Optional[bool]
            Whether to perform power interations to update the singular value estimate.

        Returns
        -------
        Array
            The transformed input.
        """
        update_singular_value_estimate = nn.merge_param(
            "update_singular_value_estimate",
            self.update_singular_value_estimate,
            update_singular_value_estimate,
        )
        layer_name = (
            type(self.layer).__name__ if self.layer_name is None else self.layer_name
        )
        params = self.param(
            layer_name, lambda *args: self.layer.init(*args)["params"], inputs
        )
        w = params[self.kernel_name]

        if self.kernel_apply_kwargs is None:
            # By default, we use the implementation in SN-GAN.
            kernel_apply = lambda x: x @ w.reshape(-1, w.shape[-1])
            in_shape = (np.prod(w.shape[:-1]),)
        else:
            # Otherwise, we extract the actual kernel transformation in the input
            # layer. This is useful for Conv2D spectral normalization in
            # [Farzan F. et al., 2019](https://arxiv.org/abs/1811.07457).
            kernel_apply = self.layer.clone(
                **self.kernel_apply_kwargs
            ).bind(  # pylint: disable=not-a-mapping
                {"params": {self.kernel_name: w}}
            )
            # Compute input shape of the kernel operator. This is correct for all
            # linear layers on Flax: Dense, Conv, Embed.
            in_shape = inputs.shape[-w.ndim + 1 : -1] + w.shape[-2:-1]

        initializing = self.is_mutable_collection("params")
        u, v = self._get_singular_vectors(initializing, kernel_apply, in_shape, w.dtype)
        u_hat, v_hat = u.value, v.value
        u_, kernel_transpose = jax.vjp(kernel_apply, v_hat)
        if update_singular_value_estimate and not initializing:
            # Run power iterations using autodiff approach inspired by
            # (Stephan H., 2020)[https://nbviewer.jupyter.org/gist/shoyer/fa9a29fd0880e2e033d7696585978bfc]).
            def scan_body(carry, _):
                u_hat, v_hat, u_ = carry
                (v_,) = kernel_transpose(u_hat)
                v_hat = _l2_normalize(v_)
                u_ = kernel_apply(v_hat)
                u_hat = _l2_normalize(u_)
                return (u_hat, v_hat, u_), None

            (u_hat, v_hat, u_), _ = jax.lax.scan(
                scan_body, (u_hat, v_hat, u_), None, length=self.iteration
            )
            u.value, v.value = u_hat, v_hat

        sigma = jnp.vdot(u_hat, u_)
        # Bound spectral norm by the `norm_multiplier`.
        sigma = jnp.maximum(sigma / self.norm_multiplier, 1.0)
        w_hat = w / jax.lax.stop_gradient(sigma)
        self.sow("intermediates", "w", w_hat)

        # Update params.
        params = flax.core.unfreeze(params)
        params[self.kernel_name] = w_hat
        layer_params = flax.core.freeze({"params": params})
        return self.layer.apply(layer_params, inputs)


class SpectralNormalizationConv2D(SpectralNormalization):
    __doc__ = (
        "Implements spectral normalization for Convolutional layers."
        "See `Generalizable Adversarial Training via Spectral Normalization <https://arxiv.org/abs/1811.07457>`_.\n"
        + "\n".join(SpectralNormalization.__doc__.split("\n")[1:])
    )

    kernel_apply_kwargs: Mapping[str, Any] = flax.core.FrozenDict(
        feature_group_count=1, padding="SAME", use_bias=False
    )


@dataclasses.dataclass
class WithSpectralConv2DNorm:
    """
    Attributes
    ----------
    spectral_norm_iteration: int
        The number of power iterations to estimate weight matrix's singular value.
    spectral_norm_bound: float
        Multiplicative constant to threshold the normalization.
        Usually under normalization, the singular value will converge to this value.
    """

    spectral_norm_iteration: int = 1.0
    spectral_norm_bound: float = 0.95

    def spectral_norm(self, layer: nn.Module, train: bool = False) -> Callable:
        return lambda *args, **kwargs: SpectralNormalizationConv2D(
            layer=layer(*args, **kwargs),
            iteration=self.spectral_norm_iteration,
            norm_multiplier=self.spectral_norm_bound,
            update_singular_value_estimate=train,
        )


@dataclasses.dataclass
class WithSpectralNorm:
    """
    Attributes
    ----------
    spectral_norm_iteration: int
        The number of power iterations to estimate weight matrix's singular value.
    spectral_norm_bound: float
        Multiplicative constant to threshold the normalization.
        Usually under normalization, the singular value will converge to this value.
    """

    spectral_norm_iteration: int = 1.0
    spectral_norm_bound: float = 0.95

    def spectral_norm(self, layer: nn.Module, train: bool = False) -> Callable:
        return lambda *args, **kwargs: SpectralNormalization(
            layer=layer(*args, **kwargs),
            iteration=self.spectral_norm_iteration,
            norm_multiplier=self.spectral_norm_bound,
            update_singular_value_estimate=train,
        )
