from fortuna.prob_model.posterior.base import PosteriorApproximator
from fortuna.prob_model.posterior.laplace import LAPLACE_NAME


class LaplacePosteriorApproximator(PosteriorApproximator):
    def __init__(self, tune_prior_log_variance: bool = True):
        """
        Laplace posterior approximator.
        """
        self.tune_prior_log_variance = tune_prior_log_variance

    def __str__(self):
        return LAPLACE_NAME
