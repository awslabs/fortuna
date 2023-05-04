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
from fortuna.prob_model.posterior.sgmcmc.cyclical_sgld import (
    CYCLICAL_SGLD_NAME,
)
from fortuna.prob_model.posterior.sgmcmc.cyclical_sgld.cyclical_sgld_approximator import (
    CyclicalSGLDPosteriorApproximator,
)
from fortuna.prob_model.posterior.sgmcmc.cyclical_sgld.cyclical_sgld_integrator import (
    cyclical_sgld_integrator,
)
from fortuna.prob_model.posterior.sgmcmc.cyclical_sgld.cyclical_sgld_state import (
    CyclicalSGLDState,
)

logger = logging.getLogger(__name__)


class CyclicalSGLDPosterior(SGMCMCPosterior):
    def __init__(
        self,
        joint: Joint,
        posterior_approximator: CyclicalSGLDPosteriorApproximator,
    ):
        """
        Cyclical Stochastic Gradient Langevin Dynamics (SGLD) approximate posterior class.

        Parameters
        ----------
        joint: Joint
            A Joint distribution object.
        posterior_approximator: CyclicalSGLDPosteriorApproximator
            A cyclical SGLD posterior approximator.
        """
        super().__init__(
            joint=joint, posterior_approximator=posterior_approximator
        )

    def __str__(self):
        return CYCLICAL_SGLD_NAME

    def get_integrator(self):
        return cyclical_sgld_integrator(
            rng_key=self.rng.get(),
            init_step_size=self.posterior_approximator.init_step_size,
            burnin_steps=self.posterior_approximator.burnin_steps,
            cycle_length=self.posterior_approximator.cycle_length,
            exploration_ratio=self.posterior_approximator.exploration_ratio,
            preconditioner=self.posterior_approximator.preconditioner,
        )

    def convert_state_from_map_state(self, *args, **kwargs):
        return CyclicalSGLDState.convert_from_map_state(*args, **kwargs)
