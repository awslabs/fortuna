import enum

from fortuna.calibration.output_calib_model.state import OutputCalibState
from fortuna.prob_model.posterior.laplace.laplace_state import LaplaceState
from fortuna.prob_model.posterior.map.map_state import MAPState
from fortuna.prob_model.posterior.normalizing_flow.advi.advi_state import \
    ADVIState
from fortuna.prob_model.posterior.state import PosteriorState
from fortuna.prob_model.posterior.swag.swag_state import SWAGState


class NameToPosteriorState(enum.Enum):
    vars()[OutputCalibState.__name__] = OutputCalibState
    vars()[PosteriorState.__name__] = PosteriorState
    vars()[MAPState.__name__] = MAPState
    vars()[ADVIState.__name__] = ADVIState
    vars()[LaplaceState.__name__] = LaplaceState
    vars()[SWAGState.__name__] = SWAGState
