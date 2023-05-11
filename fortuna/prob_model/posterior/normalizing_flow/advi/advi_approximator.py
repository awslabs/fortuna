from typing import (
    List,
    Optional,
    Tuple,
)

from fortuna.prob_model.posterior.base import PosteriorApproximator
from fortuna.prob_model.posterior.normalizing_flow.advi import ADVI_NAME


class ADVIPosteriorApproximator(PosteriorApproximator):
    def __init__(
        self,
        std_init_params: float = 0.1,
        log_std_base: float = -2.3,
        n_loss_samples: int = 3,
    ):
        """
        Automatic Differentiation Variational Inference (ADVI) approximator. It is responsible to define how the
        posterior distribution is approximated.

        Parameters
        ----------
        std_init_params : float
            The standard deviation of the Gaussian distribution used to initialize the parameters of the flow.
        log_std_base : float
            The normalizing flow transforms a base distribution into an approximation of the posterior. The base
            distribution is assumed to be an isotropic Gaussian, with this argument as the log-standard deviation.
        n_loss_samples : int
            Number of samples to approximate the loss, that is the KL divergence (or the ELBO, equivalently).
        """
        self.std_init_params = std_init_params
        self.log_std_base = log_std_base
        self.n_loss_samples = n_loss_samples

    def __str__(self):
        return ADVI_NAME
