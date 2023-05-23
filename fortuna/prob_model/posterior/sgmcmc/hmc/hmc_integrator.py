import jax
import jax.numpy as jnp

from fortuna.typing import Array
from fortuna.prob_model.posterior.sgmcmc.sgmcmc_step_schedule import (
    StepSchedule,
)
from fortuna.utils.random import generate_random_normal_like_tree
from jax._src.prng import PRNGKeyArray
from optax._src.base import PyTree
from optax import GradientTransformation
from typing import NamedTuple


class OptaxHMCState(NamedTuple):
    """Optax state for the HMC integrator."""
    count: Array
    rng_key: PRNGKeyArray
    momentum: PyTree
    params: PyTree
    hamiltonian: Array
    log_prob: Array


def hmc_integrator(
    integration_steps: int,
    rng_key: PRNGKeyArray,
    step_schedule: StepSchedule,
) -> GradientTransformation:
    """Optax implementation of the HMC integrator.

    Parameters
    ----------
        integration_steps: int
            Number of leapfrog integration steps in each trajectory.
        rng_key: PRNGKeyArray
            An initial random number generator.
        step_schedule: StepSchedule
            A function that takes training step as input and returns the step size.
    """
    def init_fn(params):
        return OptaxHMCState(
            count=jnp.zeros([], jnp.int32),
            rng_key=rng_key,
            momentum=jax.tree_util.tree_map(jnp.zeros_like, params),
            params=params,
            hamiltonian=jnp.array(-1e6, jnp.float32),
            log_prob=jnp.zeros([], jnp.float32),
        )

    def update_fn(gradient, state, params):
        step_size = step_schedule(state.count)

        def leapfrog_step():
            updates = jax.tree_map(
                lambda m: m * step_size,
                state.momentum,
            )
            momentum = jax.tree_map(
                lambda m, g: m + g * step_size,
                state.momentum,
                gradient,
            )
            return updates, OptaxHMCState(
                count=state.count + 1,
                rng_key=state.rng_key,
                momentum=momentum,
                params=state.params,
                hamiltonian=state.hamiltonian,
                log_prob=state.log_prob,
            )

        def mh_correction():
            key, new_key, uniform_key = jax.random.split(state.rng_key, 3)

            momentum = jax.tree_map(
                lambda m, g: m + g * step_size / 2,
                state.momentum,
                gradient,
            )

            momentum, _ = jax.flatten_util.ravel_pytree(momentum)
            kinetic = 0.5 * jnp.dot(momentum, momentum)
            hamiltonian = kinetic + state.log_prob
            accept_prob = jnp.minimum(1., jnp.exp(hamiltonian - state.hamiltonian))

            def _accept():
                empty_updates = jax.tree_util.tree_map(jnp.zeros_like, params)
                return empty_updates, params, hamiltonian

            def _reject():
                revert_updates = jax.tree_util.tree_map(
                    lambda sp, p: sp - p,
                    state.params,
                    params,
                )
                return revert_updates, state.params, state.hamiltonian

            updates, new_params, new_hamiltonian = jax.lax.cond(
                jax.random.uniform(uniform_key) < accept_prob,
                _accept,
                _reject,
            )

            new_momentum = generate_random_normal_like_tree(key, gradient)
            new_momentum = jax.tree_map(
                lambda m, g: m + g * step_size / 2,
                new_momentum,
                gradient,
            )

            return updates, OptaxHMCState(
                count=state.count + 1,
                rng_key=new_key,
                momentum=new_momentum,
                params=new_params,
                hamiltonian=new_hamiltonian,
                log_prob=state.log_prob,
            )

        return jax.lax.cond(
            state.count % integration_steps == 0,
            mh_correction,
            leapfrog_step,
        )

    return GradientTransformation(init_fn, update_fn)
