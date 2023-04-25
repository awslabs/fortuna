from fortuna.prob_model.posterior.base import PosteriorApproximator
from fortuna.prob_model.posterior.normalizing_flow.advi import ADVI_NAME
from typing import Optional, Tuple, List


class ADVIPosteriorApproximator(PosteriorApproximator):
    def __init__(
        self,
        std_init_params: float = 0.1,
        std_base: float = 0.1,
        n_loss_samples: int = 3,
        which_params: Optional[Tuple[List, ...]] = None
    ):
        """
        Automatic Differentiation Variational Inference (ADVI) approximator. It is responsible to define how the
        posterior distribution is approximated.

        Parameters
        ----------
        std_init_params : float
            The standard deviation of the Gaussian distribution used to initialize the parameters of the flow.
        std_base : float
            The normalizing flow transforms a base distribution into an approximation of the posterior. The base
            distribution is assumed to be an isotropic Gaussian, with this argument as standard deviation.
        n_loss_samples : int
            Number of samples to approximate the loss, that is the KL divergence (or the ELBO, equivalently).
        which_params: Optional[Tuple[List, ...]]
            Sequences of keys to the parameters of the probabilistic model for which to define the Laplace
            approximation. If `which_params` is not available, the Laplace approximation will be over all parameters.
        """
        self.std_init_params = std_init_params
        self.std_base = std_base
        self.n_loss_samples = n_loss_samples

        if which_params:
            if type(which_params) != tuple:
                raise ValueError("`which_params` must be a tuple of lists.")
            for list_keys in which_params:
                if type(list_keys) != list:
                    raise ValueError("Each element in `which_params` must be a list.")
        self.which_params = which_params

    def __str__(self):
        return ADVI_NAME
