from typing import (
    List,
    Optional,
)

from fortuna.calib_model.config.callback import Callback
from fortuna.calib_model.config.checkpointer import Checkpointer
from fortuna.calib_model.config.monitor import Monitor
from fortuna.calib_model.config.optimizer import Optimizer
from fortuna.calib_model.config.processor import Processor
from fortuna.calib_model.config.hyperparametrs import Hyperparametrs


class Config:
    def __init__(
        self,
        optimizer: Optimizer = Optimizer(),
        checkpointer: Checkpointer = Checkpointer(),
        monitor: Monitor = Monitor(),
        processor: Processor = Processor(),
        hyperparameters: Hyperparametrs = Hyperparametrs(),
        callbacks: Optional[List[Callback]] = None,
    ):
        """
        Configure the posterior distribution fitting.

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
        hyperparameters: Hyperparametrs
            It defines other hyperparameters that may be needed during model's training.
        callbacks:  Optional[List[FitCallback]]
            A list of user-defined callbacks to be called during training.
            Callbacks run sequentially in the order defined by the user.
        """
        self.optimizer = optimizer
        self.checkpointer = checkpointer
        self.monitor = monitor
        self.processor = processor
        self.hyperparameters = hyperparameters
        self.callbacks = callbacks
