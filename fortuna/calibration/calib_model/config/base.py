from fortuna.calibration.calib_model.config.checkpointer import Checkpointer
from fortuna.calibration.calib_model.config.monitor import Monitor
from fortuna.calibration.calib_model.config.optimizer import Optimizer
from fortuna.calibration.calib_model.config.processor import Processor
from fortuna.calibration.calib_model.config.callbacks import Callback
from typing import Optional, List


class Config:
    def __init__(
        self,
        optimizer: Optimizer = Optimizer(),
        checkpointer: Checkpointer = Checkpointer(),
        monitor: Monitor = Monitor(),
        processor: Processor = Processor(),
        callbacks: Optional[List[Callback]] = None
    ):
        """
        Configure the calibration of the calibration model.

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
        callbacks:  Optional[List[Callback]]
            A list of user-defined callbacks to be called during calibration.
            Callbacks run sequentially in the order defined by the user.
        """
        self.optimizer = optimizer
        self.checkpointer = checkpointer
        self.monitor = monitor
        self.processor = processor
        self.callbacks = callbacks
