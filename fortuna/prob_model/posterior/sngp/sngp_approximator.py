from typing import Optional, Dict, Any

from fortuna.prob_model.posterior.base import PosteriorApproximator
from fortuna.prob_model.posterior.sngp import SNGP_NAME


class SNGPPosteriorApproximator(PosteriorApproximator):
    def __init__(
            self,
            *args,
            output_dim: int,
            gp_hidden_features: int = 1024,
            normalize_input: bool = False,
            ridge_penalty: float = 1.0,
            momentum: Optional[float] = None,
            mean_field_factor: float = 1.0,
            **kwargs
    ):
        """
        SNGP posterior approximator. It is responsible to define how the posterior distribution is approximated.

        Parameters
        ----------
        output_dim: int
            The output dimension of the network.
        normalize_input: bool
            Whether to normalize the input using nn.LayerNorm.
        gp_hidden_features: int
            The number of random fourier features.
        ridge_penalty: float
            Initial Ridge penalty to weight covariance matrix.
            This value is used to stablize the eigenvalues of weight covariance estimate :math:`\Sigma` so that
            the matrix inverse can be computed for :math:`\Sigma = (\mathbf{I}*s+\mathbf{X}^T\mathbf{X})^{-1}`.
            The ridge factor :math:`s` cannot be too large since otherwise it will dominate
            making the covariance estimate not meaningful.
        momentum: Optional[float]
            A discount factor used to compute the moving average for posterior
            precision matrix. Analogous to the momentum factor in batch normalization.
            If `None` then update covariance matrix using a naive sum without
            momentum, which is desirable if the goal is to compute the exact
            covariance matrix by passing through data once (say in the final epoch).
            In this case, make sure to reset the precision matrix variable between
            epochs to avoid double counting.
        mean_field_factor: float
            The scale factor for mean-field approximation, used to adjust (at inference time) the influence of
            posterior variance in posterior mean approximation.
            See `Zhiyun L. et al., 2020 <https://arxiv.org/abs/2006.07584>`_ for more details.
        mean_field_factor: float
            The scale factor for mean-field approximation, used to adjust (at inference time) the influence of
            posterior variance in posterior mean approximation.
            See `Zhiyun L. et al., 2020 <https://arxiv.org/abs/2006.07584>`_ for more details.
        """
        super(SNGPPosteriorApproximator, self).__init__(*args, **kwargs)
        self.output_dim = output_dim
        self.gp_hidden_features = gp_hidden_features
        self.normalize_input = normalize_input
        self.ridge_penalty = ridge_penalty
        self.momentum = momentum
        self.mean_field_factor = mean_field_factor

    def __str__(self):
        return SNGP_NAME

    @property
    def posterior_method_kwargs(self) -> Dict[str, Any]:
        return {
            "output_dim": self.output_dim,
            "gp_hidden_features": self.gp_hidden_features,
            "normalize_input": self.normalize_input,
            "ridge_penalty": self.ridge_penalty,
            "momentum": self.momentum,
            "mean_field_factor": self.mean_field_factor,
        }
