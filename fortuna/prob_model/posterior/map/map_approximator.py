from fortuna.prob_model.posterior.base import PosteriorApproximator
from fortuna.prob_model.posterior.map import MAP_NAME


class MAPPosteriorApproximator(PosteriorApproximator):
    """Maximum-A-Posteriori posterior approximator. It is responsible to define how the posterior distribution is
    approximated."""

    def __str__(self):
        return MAP_NAME
