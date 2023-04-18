import enum

from fortuna.calibration.output_calib_model.state import OutputCalibState
from fortuna.training.train_state import TrainState


class NameToTrainState(enum.Enum):
    vars()[TrainState.__name__] = TrainState
    vars()[OutputCalibState.__name__] = OutputCalibState
