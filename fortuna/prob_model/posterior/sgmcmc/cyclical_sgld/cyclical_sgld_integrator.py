import jax
import jax.numpy as jnp

import optax
from optax._src.base import PyTree
from optax import GradientTransformation

from fortuna.typing import Array
from fortuna.prob_model.posterior.sgmcmc.sgmcmc_preconditioner import (
    PreconditionerState,
    Preconditioner,
)
from fortuna.prob_model.posterior.sgmcmc.sgmcmc_step_schedule import (
    cyclical_cosine_schedule_with_const_burnin,
)
from fortuna.prob_model.posterior.sgmcmc.sghmc.sghmc_integrator import (
    sghmc_integrator,
)
from fortuna.utils.random import generate_random_normal_like_tree
from jax._src.prng import PRNGKeyArray
from typing import Callable, NamedTuple


class OptaxCyclicalSGLDState(NamedTuple):
    """Optax state for the Cyclical SGLD integrator."""

    sgd_state: NamedTuple
    sgld_state: NamedTuple


def cyclical_sgld_integrator(
    rng_key: PRNGKeyArray,
    init_step_size: float,
    burnin_steps: int,
    cycle_length: int,
    exploration_ratio: float,
    preconditioner: Preconditioner,
) -> GradientTransformation:
    """Optax implementation of the Cyclical SGLD integrator.

    Parameters
    ----------
        rng_key: PRNGKeyArray
            An initial random number generator.
        step_schedule: StepSchedule
            A function that takes training step as input and returns the step size.
        preconditioner: Preconditioner
            See :class:`Preconditioner` for reference.
    """
    step_schedule = cyclical_cosine_schedule_with_const_burnin(
        init_step_size=init_step_size,
        burnin_steps=burnin_steps,
        cycle_length=cycle_length,
    )
    sgld = sghmc_integrator(
        momentum_decay=0.0,
        momentum_resample_steps=None,
        rng_key=rng_key,
        step_schedule=step_schedule,
        preconditioner=preconditioner,
    )
    sgd = optax.sgd(learning_rate=1.0)

    def init_fn(params):
        return OptaxCyclicalSGLDState(
            sgd_state=sgd.init(params),
            sgld_state=sgld.init(params),
        )

    def update_fn(gradient, state, *_):
        def sgd_step():
            step_size = step_schedule(state.sgld_state.count)
            preconditioner_state = preconditioner.update_preconditioner(
                gradient, state.sgld_state.preconditioner_state
            )
            new_sgld_state = state.sgld_state._replace(
                count=state.sgld_state.count + 1,
                preconditioner_state=preconditioner_state,
            )
            rescaled_gradient = jax.tree_map(
                lambda g: -1.0 * step_size * g,
                gradient,
            )
            updates, new_sgd_state = sgd.update(
                rescaled_gradient, state.sgd_state
            )
            new_state = OptaxCyclicalSGLDState(
                sgd_state=new_sgd_state,
                sgld_state=new_sgld_state,
            )
            return updates, new_state

        def sgld_step():
            updates, new_sgld_state = sgld.update(gradient, state.sgld_state)
            new_state = OptaxCyclicalSGLDState(
                sgd_state=state.sgd_state,
                sgld_state=new_sgld_state,
            )
            return updates, new_state

        updates, state = jax.lax.cond(
            ((state.sgld_state.count % cycle_length) / cycle_length)
            >= exploration_ratio,
            sgld_step,
            sgd_step,
        )
        return updates, state

    return GradientTransformation(init_fn, update_fn)
