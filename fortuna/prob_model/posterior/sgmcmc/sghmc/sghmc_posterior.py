import logging
from typing import Optional
from itertools import cycle
import pathlib

from jax._src.prng import PRNGKeyArray
from flax.core import FrozenDict
from jax import pure_callback, random

from fortuna.typing import Array
from fortuna.prob_model.joint.base import Joint
from fortuna.prob_model.joint.state import JointState
from fortuna.prob_model.posterior.sgmcmc.sgmcmc_posterior import (
    SGMCMCPosterior,
)
from fortuna.prob_model.posterior.sgmcmc.sghmc import SGHMC_NAME
from fortuna.prob_model.posterior.sgmcmc.sghmc.sghmc_approximator import (
    SGHMCPosteriorApproximator,
)
from fortuna.prob_model.posterior.sgmcmc.sghmc.sghmc_integrator import (
    sghmc_integrator,
)
from fortuna.prob_model.posterior.sgmcmc.sghmc.sghmc_state import SGHMCState

logger = logging.getLogger(__name__)


class SGHMCPosterior(SGMCMCPosterior):
    def __init__(
        self,
        joint: Joint,
        posterior_approximator: SGHMCPosteriorApproximator,
    ):
        """
        Stochastic Gradient Hamiltonian Monte Carlo approximate posterior class.

        Parameters
        ----------
        joint: Joint
            A Joint distribution object.
        posterior_approximator: SGHMCPosteriorApproximator
            A SGHMC posterior approximator.
        """
        super().__init__(
            joint=joint, posterior_approximator=posterior_approximator
        )

    def __str__(self):
        return SGHMC_NAME

    def get_integrator(self):
        return sghmc_integrator(
            momentum_decay=self.posterior_approximator.momentum_decay,
            momentum_resample_steps=None,
            rng_key=self.rng.get(),
            step_schedule=self.posterior_approximator.step_schedule,
            preconditioner=self.posterior_approximator.preconditioner,
        )

    def convert_state_from_map_state(self, *args, **kwargs):
        return SGHMCState.convert_from_map_state(*args, **kwargs)
