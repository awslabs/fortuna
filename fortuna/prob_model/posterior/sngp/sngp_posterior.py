import logging
from typing import Optional

from fortuna.data import DataLoader
from fortuna.prob_model.fit_config.base import FitConfig
from fortuna.prob_model.joint.base import Joint
from fortuna.prob_model.posterior.map.map_posterior import MAPPosterior
from fortuna.prob_model.posterior.sngp import SNGP_NAME
from fortuna.prob_model.posterior.sngp.sngp_approximator import (
    SNGPPosteriorApproximator,
)
from fortuna.prob_model.posterior.sngp.sngp_callback import ResetCovarianceCallback
from fortuna.prob_model.posterior.state import PosteriorState
from fortuna.typing import Status
from fortuna.utils.nested_dicts import find_one_path_to_key

logger = logging.getLogger(__name__)


class SNGPPosterior(MAPPosterior):
    def __init__(
        self,
        joint: Joint,
        posterior_approximator: SNGPPosteriorApproximator,
    ):
        """
        Spectral-normalized Neural Gaussian Process (`SNGP <https://arxiv.org/abs/2006.10108>`_) approximate posterior class.

        Parameters
        ----------
        joint: Joint
            A Joint distribution object.
        posterior_approximator: SNGPPosteriorApproximator
            An SNGP posterior approximator.
        """
        super().__init__(joint=joint, posterior_approximator=posterior_approximator)

    def fit(
        self,
        train_data_loader: DataLoader,
        val_data_loader: Optional[DataLoader] = None,
        fit_config: FitConfig = FitConfig(),
        **kwargs,
    ) -> Status:
        # set sngp callback to reset covariance
        callbacks = [
            ResetCovarianceCallback(
                precision_matrix_key_name="precision_matrix",
                ridge_penalty=self.joint.likelihood.model_manager.ridge_penalty,
            )
        ]
        if fit_config.callbacks is None:
            fit_config.callbacks = callbacks
        else:
            fit_config.callbacks = fit_config.callbacks + callbacks
        return super(SNGPPosterior, self).fit(
            train_data_loader, val_data_loader, fit_config, **kwargs
        )

    def __str__(self):
        return SNGP_NAME

    @staticmethod
    def _check_state(state: PosteriorState) -> None:
        path = find_one_path_to_key(state.mutable, "spectral_stats")
        if len(path) == 0:
            raise ValueError(
                "It looks like your deep feature extractor does not have Spectral Normalization, "
                "which is required by SNGP. Please include spectral normalization in your model."
                "Check out `fortuna.model.utils.spectral_norm.WithSpectralNorm` for more details."
            )
