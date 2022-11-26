from __future__ import annotations

from typing import Optional, Tuple

import jax.numpy as jnp
from jax import random
from jax._src.prng import PRNGKeyArray

from fortuna.utils.builtins import HashableMixin


class ADVIArchitecture(HashableMixin):
    def __init__(self, dim: int, std_init_params: float = 0.1):
        """
        ADVI architecture. This consists of a simple component-wise linear transformation of the input. The
        transformation includes a mean and a log-scale parameters. With this architecture, when the base distribution
        is a diagonal Gaussian, the resulting push-forward will also be. See
         [Dinh et al., 2017](https://arxiv.org/abs/1605.08803) for reference.

        :param dim: int
            Dimension of input and output of the invertible transformation.
        :param std_init_params: float
            Standard deviation of the random Gaussian initializing the parameters.
        """
        self.dim = dim
        self.std_init_params = std_init_params

    def forward(self, params: any, u: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Component-wise forward linear transformation.

        :param params: any
            Mean and log-scale parameters.
        :param u: np.ndarray
            Input to transform forward.

        :return: Tuple[jnp.ndarray, jnp.ndarray]
            v: jnp.ndarray
                Output of the forward pass.
            ldj: jnp.ndarray
                Log-determinant of the Jacobian of the forward pass.
        """
        mean, logscale = params
        return (
            mean + jnp.exp(logscale) * u,
            jnp.repeat(jnp.sum(logscale, -1), u.shape[0]),
        )

    def backward(self, params: tuple, v: jnp.ndarray) -> tuple:
        """
        Component-wise backward linear transformation.

        :param params: tuple
            Mean and log-scale parameters.
        :param v: jnp.ndarray
            Input to transform backward.

        :return: tuple
            v: jnp.ndarray
                Output of the backward pass.
            ldj: jnp.ndarray
                Log-determinant of the Jacobian of the backward pass.
        """
        mean, logscale = params
        return (
            jnp.exp(-logscale) * (v - mean),
            jnp.repeat(-jnp.sum(logscale, -1), v.shape[0]),
        )

    def init_params(
        self, rng: PRNGKeyArray, mean: Optional[jnp.ndarray] = None,
    ) -> tuple:
        """
        Initialize mean and log-scale parameters.

        :param rng: PRNGKeyArray
            Random number generator.
        :param mean: jnp.ndarray
            If the main model has already been initialized calling `model.init`, the already
            initialized parameter values can be provided here.

        :return:
            params: tuple
                Transformation parameters.
        """
        rng, key_mean, key_logscale = random.split(rng, 3)
        if mean is None:
            mean = self.std_init_params * random.normal(key_mean, (self.dim,))
        logscale = self.std_init_params * random.normal(key_logscale, (self.dim,))
        return mean, logscale
