from fortuna.prob_model.posterior.sgmcmc.base import SGMCMCPosteriorApproximator
from fortuna.prob_model.posterior.sgmcmc.cyclical_sgld import CYCLICAL_SGLD_NAME
from fortuna.prob_model.posterior.sgmcmc.sgmcmc_preconditioner import (
    Preconditioner,
    identity_preconditioner,
)


class CyclicalSGLDPosteriorApproximator(SGMCMCPosteriorApproximator):
    def __init__(
        self,
        n_samples: int = 10,
        n_thinning: int = 1,
        cycle_length: int = 1000,
        init_step_size: float = 1e-5,
        exploration_ratio: float = 0.25,
        preconditioner: Preconditioner = identity_preconditioner(),
    ) -> None:
        """
        Cyclical SGLD posterior approximator. It is responsible to define how the posterior distribution is approximated.

        The total number of available posterior samples depends on the number of training steps, `burnin_length` and
        `n_thinning` parameters, as well as `cycle_length` and `exploration_ratio`. In case if the number of training
        steps divides evenly by the cycle length, it can be calculated as follows:

        `n_cycles` = `n_training_steps` % `cycle_length`
        `n_sampling_steps` = (`n_cycles` * `cycle_length`) * (1 - `exploration_ratio`)
        `n_available_samples` = `n_sampling_steps` % `n_thinning`

        Setting the desired number of samples `n_samples` larger than `n_available_samples` will result in an
        exception.

        Parameters
        ----------
        n_samples: int
            The desired number of the posterior samples.
        n_thinning: int
            If `n_thinning` > 1, keep only each `n_thinning` sample during the sampling phase.
        cycle_length: int
            The length of each exploration/sampling cycle, in steps.
        init_step_size: float
            The initial step size.
        exploration_ratio: float
            The fraction of steps to allocate to the mode exploration phase.
        preconditioner: Preconditioner
            A `Preconditioner` instance that preconditions the approximator with information about the posterior distribution, if available.
        """
        super().__init__(
            n_samples=n_samples,
            n_thinning=n_thinning,
            preconditioner=preconditioner,
        )
        self.cycle_length = cycle_length
        self.init_step_size = init_step_size
        self.exploration_ratio = exploration_ratio

    def __str__(self) -> str:
        return CYCLICAL_SGLD_NAME
