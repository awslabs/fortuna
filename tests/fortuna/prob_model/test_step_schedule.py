import unittest

import jax.numpy as jnp

from fortuna.prob_model.posterior.sgmcmc.sgmcmc_step_schedule import (
    constant_schedule,
    constant_schedule_with_cosine_burnin,
    cosine_schedule,
    cyclical_cosine_schedule_with_const_burnin,
    polynomial_schedule,
)


class TestStepSchedule(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.count = jnp.zeros([], jnp.int32)

    def test_constant(self):
        schedule_fn = constant_schedule(init_step_size=1e-1)
        assert jnp.allclose(schedule_fn(self.count), 1e-1)
        assert jnp.allclose(schedule_fn(self.count + 1), 1e-1)

    def test_cosine(self):
        schedule_fn = cosine_schedule(init_step_size=1e-1, total_steps=10)
        assert jnp.allclose(schedule_fn(self.count), 1e-1)
        assert not jnp.allclose(schedule_fn(self.count + 1), schedule_fn(self.count))
        assert jnp.allclose(schedule_fn(self.count + 10), 0)
        assert jnp.allclose(schedule_fn(self.count + 20), 1e-1)

    def test_polynomial(self):
        schedule_fn = polynomial_schedule()
        assert schedule_fn(self.count + 1) < schedule_fn(self.count)

    def test_cosine_burnin(self):
        schedule_fn = constant_schedule_with_cosine_burnin(
            init_step_size=1e-1, final_step_size=1e-2, burnin_steps=10
        )
        assert jnp.allclose(schedule_fn(self.count), 1e-1)
        assert not jnp.allclose(schedule_fn(self.count + 1), schedule_fn(self.count))
        assert jnp.allclose(schedule_fn(self.count + 10), 1e-2)
        assert jnp.allclose(schedule_fn(self.count + 11), 1e-2)

    def test_const_burnin(self):
        schedule_fn = cyclical_cosine_schedule_with_const_burnin(
            init_step_size=1e-1, burnin_steps=10, cycle_length=10
        )
        assert jnp.allclose(schedule_fn(self.count), 1e-1)
        assert jnp.allclose(schedule_fn(self.count + 1), 1e-1)
        assert not jnp.allclose(schedule_fn(self.count + 12), 1e-1)
