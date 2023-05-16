from fortuna.prob_model.posterior.base import PosteriorApproximator
from fortuna.prob_model.posterior.sgmcmc.sgmcmc_preconditioner import (
    Preconditioner,
    identity_preconditioner,
)


class SGMCMCPosteriorApproximator(PosteriorApproximator):
    def __init__(
        self,
        n_samples: int = 10,
        n_thinning: int = 1,
        preconditioner: Preconditioner = identity_preconditioner(),
    ) -> None:
        """
        SGMCMC posterior approximator. It is responsible to define how the posterior distribution is approximated.

        Parameters
        ----------
        n_samples: int
            The desired number of the posterior samples.
        n_thinning: int
            If `n_thinning` > 1, keep only each `n_thinning` sample during the sampling phase.
        preconditioner: Preconditioner
            A `Preconditioner` instance that preconditions the approximator with information about the posterior distribution, if available.

        """
        self.n_samples = n_samples
        self.n_thinning = n_thinning
        self.preconditioner = preconditioner

    def __str__(self) -> str:
        raise NotImplementedError
