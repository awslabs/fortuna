from fortuna.calibration.output_calib_model.config.checkpointer import Checkpointer
from fortuna.calibration.output_calib_model.config.monitor import Monitor
from fortuna.calibration.output_calib_model.config.optimizer import Optimizer
from fortuna.calibration.output_calib_model.config.processor import Processor


class Config:
    def __init__(
        self,
        optimizer: Optimizer = Optimizer(),
        checkpointer: Checkpointer = Checkpointer(),
        monitor: Monitor = Monitor(),
        processor: Processor = Processor(),
    ):
        """
        Configure the calibration of the output calibration model.

        Parameters
        ----------
        optimizer: Optimizer
            It defines the optimization specifics.
        checkpointer: Checkpointer
            It handles saving and restoring checkpoints.
        monitor: Monitor
            It monitors training progress and might induce early stopping.
        processor: Processor
            It processes where computation takes place.
        """
        self.optimizer = optimizer
        self.checkpointer = checkpointer
        self.monitor = monitor
        self.processor = processor
