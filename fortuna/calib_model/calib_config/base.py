from fortuna.calib_model.calib_config.checkpointer import CalibCheckpointer
from fortuna.calib_model.calib_config.monitor import CalibMonitor
from fortuna.calib_model.calib_config.optimizer import CalibOptimizer
from fortuna.calib_model.calib_config.processor import CalibProcessor


class CalibConfig:
    def __init__(
        self,
        optimizer: CalibOptimizer = CalibOptimizer(),
        checkpointer: CalibCheckpointer = CalibCheckpointer(),
        monitor: CalibMonitor = CalibMonitor(),
        processor: CalibProcessor = CalibProcessor(),
    ):
        """
        Configure the probabilistic model calibration.

        Parameters
        ----------
        optimizer: CalibOptimizer
            It defines the optimization specifics.
        checkpointer: CalibCheckpointer
            It handles saving and restoring checkpoints.
        monitor: CalibMonitor
            It monitors training progress and might induce early stopping.
        processor: CalibProcessor
            It processes where computation takes place.
        """
        self.optimizer = optimizer
        self.checkpointer = checkpointer
        self.monitor = monitor
        self.processor = processor
