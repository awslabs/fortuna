import unittest

import jax.numpy as jnp

from fortuna.prob_model.posterior.sgmcmc.sgmcmc_preconditioner import (
    rmsprop_preconditioner,
    identity_preconditioner,
)


class TestPreconditioner(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = {"p1": jnp.zeros([1, 2], jnp.float32),
                       "p2": jnp.zeros([2, 1], jnp.float32)}
        self.grad = {"p1": jnp.ones([1, 2], jnp.float32),
                     "p2": jnp.ones([2, 1], jnp.float32)}

    def test_rmsprop(self):
        preconditioner = rmsprop_preconditioner()
        state = preconditioner.init(self.params)
        state = preconditioner.update_preconditioner(self.grad, state)
        result = preconditioner.multiply_by_m_inv(self.params, state)
        assert "p1" in result and "p2" in result
        result = preconditioner.multiply_by_m_sqrt(self.params, state)
        assert "p1" in result and "p2" in result
        result = preconditioner.multiply_by_m_sqrt_inv(self.params, state)
        assert "p1" in result and "p2" in result

    def test_identity(self):
        preconditioner = identity_preconditioner()
        state = preconditioner.init(self.params)
        state = preconditioner.update_preconditioner(self.grad, state)
        result = preconditioner.multiply_by_m_inv(self.params, state)
        assert "p1" in result and "p2" in result
        result = preconditioner.multiply_by_m_sqrt(self.params, state)
        assert "p1" in result and "p2" in result
        result = preconditioner.multiply_by_m_sqrt_inv(self.params, state)
        assert "p1" in result and "p2" in result
