from fortuna.calib_model.calib_mixin import WithCalibCheckpointingMixin
from fortuna.training.train_state_repository import TrainStateRepository


class CalibStateRepository(WithCalibCheckpointingMixin, TrainStateRepository):
    pass
