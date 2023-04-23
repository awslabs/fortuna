import enum

from fortuna.training.train_state import TrainState


class NameToTrainState(enum.Enum):
    vars()[TrainState.__name__] = TrainState
