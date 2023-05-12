from typing import Union

import jax.numpy as jnp
from jax import (
    random,
    vmap,
)
from jax._src.prng import PRNGKeyArray
from jax.scipy.stats import (
    multivariate_normal,
    norm,
)

from fortuna.distribution.base import Distribution
from fortuna.typing import Array


class DiagGaussian(Distribution):
    def __init__(self, mean: Union[float, Array], std: Union[float, Array]):
        """
        Diagonal Multivariate Gaussian class.

        :param mean: Union[float, Array]
            Mean parameter.
        :param std: Union[float, Array]
            Standard deviation parameter. If multi-dimensional, this represents the square-root of the diagonal of the
            covariance matrix.
        """
        self.mean = mean
        self.std = std
        self.dim = 1 if type(mean) in [int, float] else len(mean)

    def sample(self, rng: PRNGKeyArray, n_samples: int = 1) -> jnp.ndarray:
        """
        Sample from the diagonal Gaussian.

        :param rng: PRNGKeyArray
            Random number generator.
        :param n_samples: int
            Number of samples.

        :return: Array
            Samples. `shape = (number of samples, dimension)`
        """
        _, key = random.split(rng)
        return self.mean + self.std * random.normal(key, (n_samples, self.dim))

    def log_joint_prob(self, x: Union[float, Array]) -> Union[float, jnp.ndarray]:
        """
        Evaluate log-probability density function.

        :param x: Array
            Location(s) where the evaluation take place. Multiple locations allowed along the 0th axis.

        :return: Union[float, Array]
            Evaluation(s).
        """
        return jnp.sum(norm.logpdf(x, loc=self.mean, scale=self.std), -1)


class MultGaussian(Distribution):
    def __init__(self, mean: Array, cov: Array):
        """
        Multivariate Gaussian class.

        :param mean: Array
            Mean parameter.
        :param cov: Array
            Covariance matrix.
        """
        self.mean = mean
        self.cov = cov
        self.dim = len(mean)

    def sample(self, rng: PRNGKeyArray, n_samples: int = 1) -> jnp.ndarray:
        """
        Sample from the multivariate Gaussian.

        :param rng: PRNGKeyArray
            Random number generator.
        :param n_samples: int
            Number of samples.

        :return: Array
            Samples. `shape = (number of samples, dimension)`
        """
        keys = random.split(rng, n_samples)
        return vmap(lambda key: random.multivariate_normal(key, self.mean, self.cov))(
            keys
        )

    def log_joint_prob(self, x: Array) -> Union[float, jnp.ndarray]:
        """
        Evaluate log-probability density function.

        :param x: Array
            Location(s) where the evaluation take place. Multiple locations allowed along the 0th axis.

        :return: Union[float, Array]
            Evaluation(s).
        """
        return multivariate_normal.logpdf(x, mean=self.mean, cov=self.cov)
