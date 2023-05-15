from typing import (
    List,
    Optional,
)

from fortuna.prob_model.fit_config.callback import FitCallback
from fortuna.prob_model.fit_config.checkpointer import FitCheckpointer
from fortuna.prob_model.fit_config.hyperparameters import FitHyperparameters
from fortuna.prob_model.fit_config.monitor import FitMonitor
from fortuna.prob_model.fit_config.optimizer import FitOptimizer
from fortuna.prob_model.fit_config.processor import FitProcessor


class FitConfig:
    def __init__(
        self,
        optimizer: FitOptimizer = FitOptimizer(),
        checkpointer: FitCheckpointer = FitCheckpointer(),
        monitor: FitMonitor = FitMonitor(),
        processor: FitProcessor = FitProcessor(),
        hyperparameters: FitHyperparameters = FitHyperparameters(),
        callbacks: Optional[List[FitCallback]] = None,
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
        hyperparameters: FitHyperparameters
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
