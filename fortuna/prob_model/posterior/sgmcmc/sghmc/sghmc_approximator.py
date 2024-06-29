from typing import Union

from fortuna.prob_model.posterior.sgmcmc.base import SGMCMCPosteriorApproximator
from fortuna.prob_model.posterior.sgmcmc.sghmc import SGHMC_NAME
from fortuna.prob_model.posterior.sgmcmc.sgmcmc_preconditioner import (
    Preconditioner,
    identity_preconditioner,
)
from fortuna.prob_model.posterior.sgmcmc.sgmcmc_step_schedule import (
    StepSchedule,
    constant_schedule,
)


class SGHMCPosteriorApproximator(SGMCMCPosteriorApproximator):
    def __init__(
        self,
        n_samples: int = 10,
        n_thinning: int = 1,
        burnin_length: int = 1000,
        momentum_decay: float = 0.01,
        step_schedule: Union[StepSchedule, float] = 1e-5,
        preconditioner: Preconditioner = identity_preconditioner(),
    ) -> None:
        """
        SGHMC posterior approximator. It is responsible to define how the posterior distribution is approximated.

        The total number of available posterior samples depends on the number of training steps, `burnin_length`,
        and `n_thinning` parameters:

        `n_available_samples` = (`n_training_steps` - `burnin_length`) % `n_thinning`

        Setting the desired number of samples `n_samples` larger than `n_available_samples` will result in an
        exception.

        Parameters
        ----------
        n_samples: int
            The desired number of the posterior samples.
        n_thinning: int
            If `n_thinning` > 1, keep only each `n_thinning` sample during the sampling phase.
        burnin_length: int
            Length of the initial burn-in phase, in steps.
        momentum_decay: float
            The "friction" term that counters the noise of stochastic gradient estimates. Setting this argument to zero recovers the overamped Langevin dynamics.
        step_schedule: Union[StepSchedule, float]
            Either a constant `float` step size or a schedule function.
        preconditioner: Preconditioner
            A `Preconditioner` instance that preconditions the approximator with information about the posterior distribution, if available.

        """
        super().__init__(
            n_samples=n_samples,
            n_thinning=n_thinning,
            preconditioner=preconditioner,
        )
        if isinstance(step_schedule, float):
            step_schedule = constant_schedule(step_schedule)
        elif not callable(step_schedule):
            raise ValueError("`step_schedule` must be a a callable function.")
        self.burnin_length = burnin_length
        self.momentum_decay = momentum_decay
        self.step_schedule = step_schedule

    def __str__(self) -> str:
        return SGHMC_NAME
