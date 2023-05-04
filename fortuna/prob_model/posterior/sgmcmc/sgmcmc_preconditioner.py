import jax
import jax.numpy as jnp

from optax._src.base import PyTree
from optax import Params, GradientTransformation
from typing import NamedTuple, Callable, Any


PreconditionerState = NamedTuple


class Preconditioner(NamedTuple):
    """A sampler preconditioner class.

    Attributes
    ----------
        init: Callable
            The state initialization function.
        update_preconditioner: Callable
            The state update function that takes gradients as an input.
        multiply_by_m_sqrt: Callable
            The function that multiples its input by the square root of mass matrix :math:`\sqrt{M}`.
        multiply_by_m_inv: Callable
            The function that multiples its input by the mass matrix inverse :math:`M^{-1}`.
        multiply_by_m_sqrt_inv: Callable
            The function that multiples its input by the square root of mass matrix inverse.

    """

    init: Callable[[Params], PreconditionerState]
    update_preconditioner: Callable[
        [PyTree, PreconditionerState], PreconditionerState
    ]
    multiply_by_m_sqrt: Callable[[PyTree, PreconditionerState], PyTree]
    multiply_by_m_inv: Callable[[PyTree, PreconditionerState], PyTree]
    multiply_by_m_sqrt_inv: Callable[[PyTree, PreconditionerState], PyTree]


class RMSPropPreconditionerState(PreconditionerState):
    grad_moment_estimates: Params


def rmsprop_preconditioner(
    running_average_factor: float = 0.99, eps: float = 1.0e-7
):
    """Create an instance of the adaptive RMSProp preconditioner.

    Parameters
    ----------
        running_average_factor: float
            The decay factor for the squared gradients moving average.
        eps: float
            :math:`\epsilon` constant for numerical stability.

    Returns
    -------
        preconditioner: Preconditioner
            An instance of RMSProp preconditioner.
    """

    def init_fn(params):
        return RMSPropPreconditionerState(
            grad_moment_estimates=jax.tree_util.tree_map(
                jnp.zeros_like, params
            )
        )

    def update_preconditioner_fn(gradient, preconditioner_state):
        grad_moment_estimates = jax.tree_util.tree_map(
            lambda e, g: e * running_average_factor
            + g**2 * (1 - running_average_factor),
            preconditioner_state.grad_moment_estimates,
            gradient,
        )
        return RMSPropPreconditionerState(
            grad_moment_estimates=grad_moment_estimates
        )

    def multiply_by_m_inv_fn(vec, preconditioner_state):
        return jax.tree_util.tree_map(
            lambda e, v: v / (eps + jnp.sqrt(e)),
            preconditioner_state.grad_moment_estimates,
            vec,
        )

    def multiply_by_m_sqrt_fn(vec, preconditioner_state):
        return jax.tree_util.tree_map(
            lambda e, v: v * jnp.sqrt(eps + jnp.sqrt(e)),
            preconditioner_state.grad_moment_estimates,
            vec,
        )

    def multiply_by_m_sqrt_inv_fn(vec, preconditioner_state):
        return jax.tree_util.tree_map(
            lambda e, v: v / jnp.sqrt(eps + jnp.sqrt(e)),
            preconditioner_state.grad_moment_estimates,
            vec,
        )

    return Preconditioner(
        init=init_fn,
        update_preconditioner=update_preconditioner_fn,
        multiply_by_m_inv=multiply_by_m_inv_fn,
        multiply_by_m_sqrt=multiply_by_m_sqrt_fn,
        multiply_by_m_sqrt_inv=multiply_by_m_sqrt_inv_fn,
    )


class IdentityPreconditionerState(PreconditionerState):
    pass


def identity_preconditioner():
    """Create an instance of no-op identity preconditioner.

    Returns
    -------
    preconditioner: Preconditioner
        An instance of identity preconditioner.
    """

    def init_fn(_):
        return IdentityPreconditionerState()

    def update_preconditioner_fn(*args, **kwargs):
        return IdentityPreconditionerState()

    def multiply_by_m_inv_fn(vec, _):
        return vec

    def multiply_by_m_sqrt_fn(vec, _):
        return vec

    def multiply_by_m_sqrt_inv_fn(vec, _):
        return vec

    return Preconditioner(
        init=init_fn,
        update_preconditioner=update_preconditioner_fn,
        multiply_by_m_inv=multiply_by_m_inv_fn,
        multiply_by_m_sqrt=multiply_by_m_sqrt_fn,
        multiply_by_m_sqrt_inv=multiply_by_m_sqrt_inv_fn,
    )
