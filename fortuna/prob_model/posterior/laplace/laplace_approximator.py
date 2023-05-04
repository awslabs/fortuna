from fortuna.prob_model.posterior.base import PosteriorApproximator
from fortuna.prob_model.posterior.laplace import LAPLACE_NAME


class LaplacePosteriorApproximator(PosteriorApproximator):
    def __init__(self):
        """
        Laplace posterior approximator.
        """

    def __str__(self):
        return LAPLACE_NAME
