import numpy as np
import jax.numpy as jnp
from fortuna.typing import Array

from typing import Callable

StepSchedule = Callable[[Array], Array]


def constant_schedule(init_step_size: float) -> StepSchedule:
    """Create a constant step schedule.

    Parameters
    ----------
    init_step_size: float
        The step size.

    Returns
    -------
    schedule_fn: StepSchedule
    """

    def schedule(step: Array):
        return init_step_size

    return schedule


def cosine_schedule(init_step_size: float, total_steps: int) -> StepSchedule:
    """Create a cosine step schedule.

    Parameters
    ----------
    init_step_size: float
        The initial step size.
    total_steps: int
        The cycle length, in steps.

    Returns
    -------
    schedule_fn: StepSchedule
    """

    def schedule(step: Array):
        t = step / total_steps
        return 0.5 * init_step_size * (1 + jnp.cos(t * np.pi))

    return schedule


def polynomial_schedule(a: float = 1., b: float = 1., gamma: float = 0.55) -> StepSchedule:
    """Create a polynomial step schedule.

    Parameters
    ----------
    a: float
        Scale of all step sizes.
    b: float
        The stabilization constant.
    gamma: float
        The decay rate :math:`\gamma \in (0.5, 1.0]`.

    Returns
    -------
    schedule_fn: StepSchedule
    """

    def schedule(step: Array):
        return a * (b + step) ** (- gamma)

    return schedule


def constant_schedule_with_cosine_burnin(
    init_step_size: float, final_step_size: float, burnin_steps: int
) -> StepSchedule:
    """Create a constant schedule with cosine burn-in.

    Parameters
    ----------
    init_step_size: float
        The initial step size.
    final_step_size: float
        The desired final step size.
    burnin_steps: int
        The length of burn-in, in steps.

    Returns
    -------
    schedule_fn: StepSchedule
    """

    def schedule(step: Array):
        t = jnp.minimum(step / burnin_steps, 1.0)
        coef = (1 + jnp.cos(t * np.pi)) * 0.5
        return coef * init_step_size + (1 - coef) * final_step_size

    return schedule


def cyclical_cosine_schedule_with_const_burnin(
    init_step_size: float, burnin_steps: int, cycle_length: int
) -> StepSchedule:
    """Create a cyclical cosine schedule with constant burn-in.

    Parameters
    ----------
    init_step_size: float
        The initial step size.
    burnin_steps: int
        The length of burn-in, in steps.
    cycle_length: int
        The length of the cosine cycle, in steps.

    Returns
    -------
    schedule_fn: StepSchedule
    """

    def schedule(step: Array):
        t = jnp.maximum(step - burnin_steps - 1, 0.0)
        t = (t % cycle_length) / cycle_length
        return 0.5 * init_step_size * (1 + jnp.cos(t * np.pi))

    return schedule
