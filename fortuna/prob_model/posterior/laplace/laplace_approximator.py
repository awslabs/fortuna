from typing import List, Optional, Tuple

from fortuna.prob_model.posterior.base import PosteriorApproximator
from fortuna.prob_model.posterior.laplace import LAPLACE_NAME


class LaplacePosteriorApproximator(PosteriorApproximator):
    def __init__(self, which_params: Optional[Tuple[List, ...]] = None):
        """
        Laplace posterior approximator. It is responsible to define how the posterior distribution is approximated.

        Parameters
        ----------
        which_params: Optional[Tuple[List, ...]]
            Sequences of keys to the parameters of the probabilistic model for which to define the Laplace
            approximation. If `which_params` is not available, the Laplace approximation will be over all parameters.
        """
        if which_params:
            if not isinstance(which_params, tuple):
                raise ValueError("`which_params` must be a tuple of lists.")
            for list_keys in which_params:
                if not isinstance(list_keys, list):
                    raise ValueError("Each element in `which_params` must be a list.")
        self.which_params = which_params

    def __str__(self):
        return LAPLACE_NAME
