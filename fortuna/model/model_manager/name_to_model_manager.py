import enum

from fortuna.model.model_manager.classification import (
    ClassificationModelManager,
    SNGPClassificationModelManager,
)
from fortuna.prob_model.posterior.deep_ensemble import DEEP_ENSEMBLE_NAME
from fortuna.prob_model.posterior.laplace import LAPLACE_NAME
from fortuna.prob_model.posterior.map import MAP_NAME
from fortuna.prob_model.posterior.normalizing_flow.advi import ADVI_NAME
from fortuna.prob_model.posterior.sgmcmc.cyclical_sgld import CYCLICAL_SGLD_NAME
from fortuna.prob_model.posterior.sgmcmc.sghmc import SGHMC_NAME
from fortuna.prob_model.posterior.sngp import SNGP_NAME
from fortuna.prob_model.posterior.swag import SWAG_NAME


class ClassificationModelManagers(enum.Enum):
    """Map approximator name to model manager classes"""

    vars()[MAP_NAME] = ClassificationModelManager
    vars()[ADVI_NAME] = ClassificationModelManager
    vars()[DEEP_ENSEMBLE_NAME] = ClassificationModelManager
    vars()[LAPLACE_NAME] = ClassificationModelManager
    vars()[SWAG_NAME] = ClassificationModelManager
    vars()[SNGP_NAME] = SNGPClassificationModelManager
    vars()[SGHMC_NAME] = ClassificationModelManager
    vars()[CYCLICAL_SGLD_NAME] = ClassificationModelManager
