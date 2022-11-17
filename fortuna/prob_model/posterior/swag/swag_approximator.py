from fortuna.prob_model.posterior.base import PosteriorApproximator
from fortuna.prob_model.posterior.swag import SWAG_NAME


class SWAGPosteriorApproximator(PosteriorApproximator):
    def __init__(self, rank: int = 5):
        """
        SWAG posterior approximator. It is responsible to define how the posterior distribution is approximated.

        Parameters
        ----------
        rank: int
            SWAG approximates the posterior with a Gaussian distribution. The Gaussian's covariance matrix is formed by
            a diagonal matrix, and a low-rank empirical approximation. This argument defines the rank of the low-rank
            empirical covariance approximation. It must be at least 2.
        """
        self.rank = rank

    def __str__(self):
        return SWAG_NAME
