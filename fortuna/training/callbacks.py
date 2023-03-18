from fortuna.training.train_state import TrainState


class Callback:
    """
    Base class to define new callback functions. To define a new callback, create a child of this class and
    override the relevant methods.
    """
    def training_epoch_start(self, state: TrainState) -> TrainState:
        """Called at the beginning of every training epoch"""
        return state

    def training_epoch_end(self, state: TrainState) -> TrainState:
        """Called at the end of every training epoch"""
        return state

    def training_step_end(self, state: TrainState) -> TrainState:
        """Called after every minibatch update"""
        return state
