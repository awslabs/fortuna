from functools import partial
import unittest

from jax import (
    value_and_grad,
    vmap,
)
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import numpy as np

from fortuna.prob_model.posterior.sgmcmc.sgmcmc_diagnostic import (
    effective_sample_size,
    kernel_stein_discrepancy_imq,
)

DATA_SIZE = 1000


class TestDiagnostic(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rng = np.random.default_rng(0)

        self.mu = np.array([0.0, 0.0])
        self.sigma = np.array([[1.5, 0.5], [0.5, 1.5]])

        def _mvn_log_density(params, mu=self.mu, sigma=self.sigma):
            diff = params - mu
            log_density = -jnp.log(2 * jnp.pi) * mu.size / 2
            log_density -= jnp.log(jnp.linalg.det(sigma)) / 2
            log_density -= diff.T @ jnp.linalg.inv(sigma) @ diff / 2
            return log_density

        self.mvn_log_density_grad = vmap(value_and_grad(_mvn_log_density))

    def unflatten(self, x, keys=("x", "y")):
        assert len(x.shape) == 2 and x.shape[-1] == len(keys)
        return [{k: v for k, v in zip(keys, val)} for val in x]

    def test_ksd_imq(self):
        samp1_flat = self.rng.multivariate_normal(self.mu, self.sigma, size=DATA_SIZE)
        samp2_flat = self.rng.multivariate_normal(
            self.mu, self.sigma**3, size=DATA_SIZE
        )
        _, grad1_flat = self.mvn_log_density_grad(samp1_flat)
        _, grad2_flat = self.mvn_log_density_grad(samp2_flat)
        assert kernel_stein_discrepancy_imq(
            samp1_flat, grad1_flat
        ) < kernel_stein_discrepancy_imq(samp2_flat, grad2_flat)
        samp1_tree = self.unflatten(samp1_flat)
        grad1_tree = self.unflatten(grad1_flat)
        assert jnp.allclose(
            kernel_stein_discrepancy_imq(samp1_flat, grad1_flat),
            kernel_stein_discrepancy_imq(samp1_tree, grad1_tree),
        )

    def test_ess(self):
        samp1_flat = self.rng.multivariate_normal(self.mu, self.sigma, size=DATA_SIZE)
        ess = effective_sample_size(samp1_flat)
        assert samp1_flat.shape[-1] == ess.shape[0]
        assert jnp.alltrue(0 <= ess) and jnp.alltrue(ess <= len(samp1_flat))
        samp1_tree = self.unflatten(samp1_flat)
        ess_tree = effective_sample_size(samp1_tree)
        vals, _unravel_fn = ravel_pytree(ess_tree)
        assert jnp.allclose(vals, ess)
