import enum

from fortuna.prob_model.posterior.deep_ensemble import DEEP_ENSEMBLE_NAME
from fortuna.prob_model.posterior.deep_ensemble.deep_ensemble_posterior import \
    DeepEnsemblePosterior
from fortuna.prob_model.posterior.laplace import LAPLACE_NAME
from fortuna.prob_model.posterior.laplace.laplace_posterior import \
    LaplacePosterior
from fortuna.prob_model.posterior.map import MAP_NAME
from fortuna.prob_model.posterior.map.map_posterior import MAPPosterior
from fortuna.prob_model.posterior.normalizing_flow.advi import ADVI_NAME
from fortuna.prob_model.posterior.normalizing_flow.advi.advi_posterior import \
    ADVIPosterior
from fortuna.prob_model.posterior.swag import SWAG_NAME
from fortuna.prob_model.posterior.swag.swag_posterior import SWAGPosterior


class PosteriorApproximations(enum.Enum):
    """Map approximator name to posterior posterior_approximation."""

    vars()[MAP_NAME] = MAPPosterior
    vars()[ADVI_NAME] = ADVIPosterior
    vars()[DEEP_ENSEMBLE_NAME] = DeepEnsemblePosterior
    vars()[LAPLACE_NAME] = LaplacePosterior
    vars()[SWAG_NAME] = SWAGPosterior
