from __future__ import annotations

from typing import Optional, Tuple, Dict

import jax.numpy as jnp
from jax import random
from jax._src.prng import PRNGKeyArray

from fortuna.utils.builtins import HashableMixin


class ADVIArchitecture(HashableMixin):
    def __init__(self, dim: int, std_init_params: float = 0.1):
        """
        ADVI architecture. This consists of a simple component-wise linear transformation of the input. The
        transformation includes a mean and a log-std parameters. With this architecture, when the base distribution
        is a diagonal Gaussian, the resulting push-forward will also be. See
         [Dinh et al., 2017](https://arxiv.org/abs/1605.08803) for reference.

        Parameters
        ----------
        dim: int
            Dimension of input and output of the invertible transformation.
        std_init_params: float
            Standard deviation of the random Gaussian initializing the parameters.
        """
        self.dim = dim
        self.std_init_params = std_init_params

    def forward(self, params: Dict[str, jnp.ndarray], u: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Component-wise forward linear transformation.

        Parameters
        ----------
        params: Dict[str, jnp.ndarray]
            Mean and log-std parameters.
        u: jnp.ndarray
            Input to transform forward.

        Returns
        -------
        Tuple[jnp.ndarray, jnp.ndarray]
            v: jnp.ndarray
                Output of the forward pass.
            ldj: jnp.ndarray
                Log-determinant of the Jacobian of the forward pass.
        """
        return (
            params["mean"] + jnp.exp(params["log_std"]) * u,
            jnp.repeat(jnp.sum(params["log_std"], -1), u.shape[0]),
        )

    def backward(self, params: Dict[str, jnp.ndarray], v: jnp.ndarray) -> Tuple[jnp.array, jnp.array]:
        """
        Component-wise backward linear transformation.

        Parameters
        ----------
        params: Dict[str, jnp.ndarray]
            Mean and log-std parameters.
        v: jnp.ndarray
            Input to transform backward.

        Returns
        -------
        Tuple[jnp.array, jnp.array]
            v: jnp.ndarray
                Output of the backward pass.
            ldj: jnp.ndarray
                Log-determinant of the Jacobian of the backward pass.
        """
        return (
            jnp.exp(-params["log_std"]) * (v - params["mean"]),
            jnp.repeat(-jnp.sum(params["log_std"], -1), v.shape[0]),
        )

    def init_params(
        self,
        rng: PRNGKeyArray,
        mean: Optional[jnp.ndarray] = None,
        log_std: Optional[jnp.ndarray] = None
    ) -> Dict[str, jnp.ndarray]:
        """
        Initialize architecture parameters.

        Parameters
        ----------
        rng: PRNGKeyArray
            Random number generator.
        mean: jnp.ndarray
            If the main model has already been initialized calling `model.init`, the already
            initialized mean parameter values can be provided here.
        log_std: jnp.ndarray
            If the main model has already been initialized calling `model.init`, the already
            initialized log-std parameter values can be provided here.

        Returns
        -------
        Tuple[jnp.array, jnp.array]
            Transformation parameters.
        """
        rng, key_mean, key_log_std = random.split(rng, 3)
        if mean is None:
            mean = self.std_init_params * random.normal(key_mean, (self.dim,))
        if log_std is None:
            log_std = self.std_init_params * random.normal(key_log_std, (self.dim,))
        return dict(mean=mean, log_std=log_std)
