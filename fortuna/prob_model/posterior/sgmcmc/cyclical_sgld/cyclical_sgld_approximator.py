from typing import Union, Optional

from fortuna.prob_model.posterior.base import PosteriorApproximator
from fortuna.prob_model.posterior.sgmcmc.sgmcmc_preconditioner import (
    Preconditioner,
    identity_preconditioner,
)
from fortuna.prob_model.posterior.sgmcmc.cyclical_sgld import (
    CYCLICAL_SGLD_NAME,
)


class CyclicalSGLDPosteriorApproximator(PosteriorApproximator):
    def __init__(
        self,
        num_samples: int = 10,
        num_thinning: Optional[int] = 10,
        init_step_size: float = 1e-5,
        burnin_steps: int = 0,
        cycle_length: int = 1000,
        exploration_ratio: float = 0.25,
        preconditioner: Preconditioner = identity_preconditioner(),
    ) -> None:
        """
        Cyclical SGLD posterior approximator. It is responsible to define how the posterior distribution is approximated.

        Parameters
        ----------
        num_samples: int
            The desired number of the posterior samples.
        num_thinning: int
            If `num_thinning` is not `None`, keep only each `num_thinning` sample during the sampling phase.
        init_step_size: float
            The initial step size.
        burnin_steps: int
            The number of steps for the initial burn-in phase.
        cycle_length: int
            The length of each training cycle, in steps.
        exploration_ratio: float
            The fraction of steps to allocate to the mode exploration phase.

        preconditioner: Preconditioner
            A `Preconditioner` instance that preconditions the approximator with information about the posterior distribution, if available.

        """
        self.num_samples = num_samples
        self.num_thinning = num_thinning
        self.init_step_size = init_step_size
        self.burnin_steps = burnin_steps
        self.cycle_length = cycle_length
        self.exploration_ratio = exploration_ratio
        self.preconditioner = preconditioner

    def __str__(self) -> str:
        return CYCLICAL_SGLD_NAME
