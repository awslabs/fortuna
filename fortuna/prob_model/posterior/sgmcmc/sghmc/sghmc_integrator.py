import jax
import jax.numpy as jnp

from fortuna.typing import Array
from fortuna.prob_model.posterior.sgmcmc.sgmcmc_preconditioner import (
    PreconditionerState,
    Preconditioner,
)
from fortuna.prob_model.posterior.sgmcmc.sgmcmc_step_schedule import (
    StepSchedule,
)
from fortuna.utils.random import generate_random_normal_like_tree
from jax._src.prng import PRNGKeyArray
from optax._src.base import PyTree
from optax import GradientTransformation
from typing import NamedTuple, Optional


class OptaxSGHMCState(NamedTuple):
    """Optax state for the SGHMC integrator."""

    count: Array
    rng_key: PRNGKeyArray
    momentum: PyTree
    preconditioner_state: PreconditionerState


def sghmc_integrator(
    momentum_decay: float,
    momentum_resample_steps: Optional[int],
    rng_key: PRNGKeyArray,
    step_schedule: StepSchedule,
    preconditioner: Preconditioner,
) -> GradientTransformation:
    """Optax implementation of the SGHMC integrator.

    Parameters
    ----------
        momentum_decay: float
            The momentum decay parameter.
        rng_key: PRNGKeyArray
            An initial random number generator.
        step_schedule: StepSchedule
            A function that takes training step as input and returns the step size.
        preconditioner: Preconditioner
            See :class:`Preconditioner` for reference.
    """
    # Implementation was partually adapted from https://github.com/google-research/google-research/blob/master/bnn_hmc/core/sgmcmc.py#L56

    def init_fn(params):
        return OptaxSGHMCState(
            count=jnp.zeros([], jnp.int32),
            rng_key=rng_key,
            momentum=jax.tree_util.tree_map(jnp.zeros_like, params),
            preconditioner_state=preconditioner.init(params),
        )

    def update_fn(gradient, state, *_):
        step_size = step_schedule(state.count)

        preconditioner_state = preconditioner.update_preconditioner(
            gradient, state.preconditioner_state
        )

        key, new_key = jax.random.split(state.rng_key)
        noise = generate_random_normal_like_tree(key, gradient)
        noise = preconditioner.multiply_by_m_sqrt(noise, preconditioner_state)

        momentum = jax.lax.cond(
            momentum_resample_steps is not None
            and state.count % momentum_resample_steps == 0,
            lambda: jax.tree_util.tree_map(
                jnp.zeros_like, gradient
            ),
            lambda: state.momentum,
        )

        momentum = jax.tree_map(
            lambda m, g, n: momentum_decay * m
            + g * jnp.sqrt(step_size)
            + n * jnp.sqrt(2 * (1 - momentum_decay)),
            momentum,
            gradient,
            noise,
        )
        updates = preconditioner.multiply_by_m_inv(
            momentum, preconditioner_state
        )
        updates = jax.tree_map(lambda m: m * jnp.sqrt(step_size), updates)
        return updates, OptaxSGHMCState(
            count=state.count + 1,
            rng_key=new_key,
            momentum=momentum,
            preconditioner_state=preconditioner_state,
        )

    return GradientTransformation(init_fn, update_fn)
