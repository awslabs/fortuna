import unittest

from jax import numpy as jnp
from jax.flatten_util import ravel_pytree

from fortuna.prob_model.prior import (DiagonalGaussianPrior,
                                      IsotropicGaussianPrior)
from fortuna.utils.random import RandomNumberGenerator


class TestIsotropicDiagGaussianPrior(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_var = 0.0
        self.prior = IsotropicGaussianPrior(log_var=self.log_var)
        self.prior.rng = RandomNumberGenerator(seed=0)
        self.params = dict(model=jnp.arange(3), lik_log_var=jnp.arange(4, 7))

    def test_log_joint_prob(self):
        assert jnp.array([self.prior.log_joint_prob(self.params)]).shape == (1,)
        assert jnp.allclose(
            self.prior.log_joint_prob(jnp.zeros(2)),
            -(jnp.log(2 * jnp.pi) + self.log_var),
        )

    def test_sample(self):
        n_params = len(ravel_pytree(self.params)[0])
        rav_samples = ravel_pytree(self.prior.sample(self.params))[0]
        assert rav_samples.size == n_params


class TestDiagGaussianPrior(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_var = 0.1 + jnp.arange(-2, 4)
        self.prior = DiagonalGaussianPrior(log_var=self.log_var)
        self.prior.rng = RandomNumberGenerator(seed=0)
        self.params = dict(model=jnp.arange(3), lik_log_var=jnp.arange(4, 7))
        self.n_samples = 3

    def test_log_joint_prob(self):
        assert jnp.array([self.prior.log_joint_prob(self.params)]).shape == (1,)
        assert jnp.allclose(
            self.prior.log_joint_prob(jnp.zeros(len(self.log_var))),
            -0.5 * jnp.sum(jnp.log(2 * jnp.pi) + self.log_var),
        )

    def test_sample(self):
        n_params = len(ravel_pytree(self.params)[0])
        rav_samples = ravel_pytree(self.prior.sample(self.params))[0]
        assert rav_samples.size == n_params
