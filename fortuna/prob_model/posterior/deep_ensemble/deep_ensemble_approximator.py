from fortuna.prob_model.posterior.base import PosteriorApproximator
from fortuna.prob_model.posterior.deep_ensemble import DEEP_ENSEMBLE_NAME


class DeepEnsemblePosteriorApproximator(PosteriorApproximator):
    def __init__(self, ensemble_size: int = 5):
        """
        Deep ensemble posterior approximator. It is responsible to define how the posterior distribution is
        approximated.

        Parameters
        ----------
        ensemble_size : int
            The size of the ensemble.
        """
        self.ensemble_size = ensemble_size

    def __str__(self):
        return DEEP_ENSEMBLE_NAME
