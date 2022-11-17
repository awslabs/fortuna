from typing import Optional

import jax.numpy as jnp
from fortuna.prob_model.prior.base import Prior
from fortuna.typing import Params
from jax import random
from jax._src.prng import PRNGKeyArray
from jax.flatten_util import ravel_pytree


class IsotropicGaussianPrior(Prior):
    def __init__(self, log_var: Optional[float] = 0.0):
        """
        A diagonal Gaussian prior class.

        Parameters
        ----------
        log_var : Optional[float]
            Prior log-variance value. The covariance matrix of the prior distribution is given by a diagonal matrix
            with this parameter on every entry of the diagonal.
        """
        super().__init__()
        self.log_var = log_var
        self.prec = jnp.exp(-self.log_var)
        self.std = jnp.exp(0.5 * self.log_var)
        self.log2pi = jnp.log(2 * jnp.pi)

    def log_prob(self, params: Params) -> float:
        rav = ravel_pytree(params)[0]
        n = len(rav)
        return -0.5 * (self.prec * jnp.sum(rav ** 2) + n * (self.log2pi + self.log_var))

    def sample(self, params_like: Params, rng: Optional[PRNGKeyArray] = None) -> Params:
        dummy_rav, unravel = ravel_pytree(params_like)
        n = len(dummy_rav)
        if rng is None:
            rng = self.rng.get()
        rav_samples = self.std * random.normal(rng, shape=(n,))
        return unravel(rav_samples)


class DiagonalGaussianPrior(Prior):
    def __init__(self, log_var: jnp.ndarray):
        """
        A diagonal Gaussian prior class.

        Parameters
        ----------
        log_var : jnp.ndarray
            Prior log-variance vector corresponding to the logarithm of the diagonal of the prior covariance matrix.
        """
        super().__init__()
        self.log_var = log_var
        self.log2pi = jnp.log(2 * jnp.pi)

    def log_prob(self, params: Params) -> float:
        rav = ravel_pytree(params)[0]
        return -0.5 * jnp.sum(
            jnp.exp(-self.log_var) * rav ** 2 + self.log2pi + self.log_var
        )

    def sample(self, params_like: Params, rng: Optional[PRNGKeyArray] = None) -> Params:
        dummy_rav, unravel = ravel_pytree(params_like)
        n = len(dummy_rav)
        if rng is None:
            rng = self.rng.get()
        rav_samples = jnp.exp(0.5 * self.log_var) * random.normal(rng, shape=(n,))
        return unravel(rav_samples)
