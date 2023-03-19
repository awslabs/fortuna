from typing import List, Optional

from fortuna.prob_model.fit_config.checkpointer import FitCheckpointer
from fortuna.prob_model.fit_config.monitor import FitMonitor
from fortuna.prob_model.fit_config.optimizer import FitOptimizer
from fortuna.prob_model.fit_config.processor import FitProcessor
from fortuna.prob_model.fit_config.callbacks import Callback


class FitConfig:
    def __init__(
        self,
        optimizer: FitOptimizer = FitOptimizer(),
        checkpointer: FitCheckpointer = FitCheckpointer(),
        monitor: FitMonitor = FitMonitor(),
        processor: FitProcessor = FitProcessor(),
        callbacks: Optional[List[Callback]] = None
    ):
        """
        Configure the posterior distribution fitting.

        Parameters
        ----------
        optimizer: FitOptimizer
            It defines the optimization specifics.
        checkpointer: FitCheckpointer
            It handles saving and restoring checkpoints.
        monitor: FitMonitor
            It monitors training progress and might induce early stopping.
        processor: FitProcessor
            It processes where computation takes place.
        callbacks:  Optional[List[Callback]]
            A list of user-defined callbacks to be called during training.
            Callbacks run sequentially in the order defined by the user.
        """
        self.optimizer = optimizer
        self.checkpointer = checkpointer
        self.monitor = monitor
        self.processor = processor
        self.callbacks = callbacks
