from fortuna.output_calib_model.output_calib_mixin import (
    WithOutputCalibCheckpointingMixin,
)
from fortuna.training.train_state_repository import TrainStateRepository


class OutputCalibStateRepository(
    WithOutputCalibCheckpointingMixin, TrainStateRepository
):
    pass
