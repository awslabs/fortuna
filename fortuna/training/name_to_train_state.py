import enum

from fortuna.training.calib_state import CalibState
from fortuna.training.train_state import TrainState


class NameToTrainState(enum.Enum):
    vars()[TrainState.__name__] = TrainState
    vars()[CalibState.__name__] = CalibState
