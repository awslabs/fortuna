from typing import Union

from fortuna.prob_model.posterior.base import PosteriorApproximator
from fortuna.prob_model.posterior.sgmcmc.sgmcmc_preconditioner import (
    Preconditioner,
    identity_preconditioner,
)
from fortuna.prob_model.posterior.sgmcmc.sgmcmc_step_schedule import (
    StepSchedule,
    constant_schedule,
)
from fortuna.prob_model.posterior.sgmcmc.sghmc import SGHMC_NAME


class SGHMCPosteriorApproximator(PosteriorApproximator):
    def __init__(
        self,
        num_samples: int = 10,
        momentum_decay: float = 0.01,
        step_schedule: Union[StepSchedule, float] = 1e-5,
        preconditioner: Preconditioner = identity_preconditioner(),
    ) -> None:
        """
        SGHMC posterior approximator. It is responsible to define how the posterior distribution is approximated.

        Parameters
        ----------
        num_samples: int
            The desired number of the posterior samples.
        momentum_decay: float
            The "friction" term that counters the noise of stochastic gradient estimates. Setting this argument to zero recovers the overamped Langevin dynamics.
        step_schedule: Union[StepSchedule, float]
            Either a constant `float` step size or a schedule function.
        preconditioner: Preconditioner
            A `Preconditioner` instance that preconditions the approximator with information about the posterior distribution, if available.

        """

        if isinstance(step_schedule, float):
            step_schedule = constant_schedule(step_schedule)
        elif not callable(step_schedule):
            raise ValueError(f"`step_schedule` must be a a callable function.")
        self.num_samples = num_samples
        self.momentum_decay = momentum_decay
        self.step_schedule = step_schedule
        self.preconditioner = preconditioner

    def __str__(self) -> str:
        return SGHMC_NAME
