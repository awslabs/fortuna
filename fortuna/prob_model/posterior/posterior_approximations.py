import enum

from fortuna.prob_model.posterior.deep_ensemble import DEEP_ENSEMBLE_NAME
from fortuna.prob_model.posterior.deep_ensemble.deep_ensemble_posterior import (
    DeepEnsemblePosterior,
)
from fortuna.prob_model.posterior.laplace import LAPLACE_NAME
from fortuna.prob_model.posterior.laplace.laplace_posterior import LaplacePosterior
from fortuna.prob_model.posterior.map import MAP_NAME
from fortuna.prob_model.posterior.map.map_posterior import MAPPosterior
from fortuna.prob_model.posterior.normalizing_flow.advi import ADVI_NAME
from fortuna.prob_model.posterior.normalizing_flow.advi.advi_posterior import (
    ADVIPosterior,
)
from fortuna.prob_model.posterior.sgmcmc.cyclical_sgld import CYCLICAL_SGLD_NAME
from fortuna.prob_model.posterior.sgmcmc.cyclical_sgld.cyclical_sgld_posterior import (
    CyclicalSGLDPosterior,
)
from fortuna.prob_model.posterior.sgmcmc.sghmc import SGHMC_NAME
from fortuna.prob_model.posterior.sgmcmc.sghmc.sghmc_posterior import SGHMCPosterior
from fortuna.prob_model.posterior.sngp import SNGP_NAME
from fortuna.prob_model.posterior.sngp.sngp_posterior import SNGPPosterior
from fortuna.prob_model.posterior.swag import SWAG_NAME
from fortuna.prob_model.posterior.swag.swag_posterior import SWAGPosterior


class PosteriorApproximations(enum.Enum):
    """Map approximator name to posterior posterior_approximation."""

    vars()[MAP_NAME] = MAPPosterior
    vars()[ADVI_NAME] = ADVIPosterior
    vars()[DEEP_ENSEMBLE_NAME] = DeepEnsemblePosterior
    vars()[LAPLACE_NAME] = LaplacePosterior
    vars()[SWAG_NAME] = SWAGPosterior
    vars()[SNGP_NAME] = SNGPPosterior
    vars()[SGHMC_NAME] = SGHMCPosterior
    vars()[CYCLICAL_SGLD_NAME] = CyclicalSGLDPosterior
