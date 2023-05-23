from typing import Union

from fortuna.prob_model.posterior.sgmcmc.base import (
    SGMCMCPosteriorApproximator,
)
from fortuna.prob_model.posterior.sgmcmc.sgmcmc_preconditioner import (
    Preconditioner,
    identity_preconditioner,
)
from fortuna.prob_model.posterior.sgmcmc.sgmcmc_step_schedule import (
    StepSchedule,
    constant_schedule,
)
from fortuna.prob_model.posterior.sgmcmc.hmc import HMC_NAME


class HMCPosteriorApproximator(SGMCMCPosteriorApproximator):
    def __init__(
        self,
        n_samples: int = 10,
        n_thinning: int = 1,
        burnin_length: int = 1000,
        integration_steps: int = 50_000,
        step_schedule: Union[StepSchedule, float] = 3e-5,
    ) -> None:
        """
        HMC posterior approximator. It is responsible to define how the posterior distribution is approximated.

        Parameters
        ----------
        n_samples: int
            The desired number of the posterior samples.
        n_thinning: int
            If `n_thinning` > 1, keep only each `n_thinning` sample during the sampling phase.
        burnin_length: int
            Length of the initial burn-in phase, in steps.
        integration_steps: int
            Number of integration steps per trajectory.
        step_schedule: Union[StepSchedule, float]
            Either a constant `float` step size or a schedule function.

        """
        super().__init__(
            n_samples=n_samples,
            n_thinning=n_thinning,
        )
        if isinstance(step_schedule, float):
            step_schedule = constant_schedule(step_schedule)
        elif not callable(step_schedule):
            raise ValueError(f"`step_schedule` must be a a callable function.")
        self.burnin_length = burnin_length
        self.integration_steps = integration_steps
        self.step_schedule = step_schedule

    def __str__(self) -> str:
        return HMC_NAME
