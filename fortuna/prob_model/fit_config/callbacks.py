from fortuna.training.train_state import TrainState


class Callback:
    """
    Base class to define new callback functions. To define a new callback, create a child of this class and
    override the relevant methods.

    Example
    -------
    The following is a custom callback that prints the number of model's parameters at the start of each epoch.

    .. code-block:: python

        class CountParamsCallback(Callback):
            def training_epoch_start(self, state: TrainState) -> TrainState:
                params, unravel = ravel_pytree(state.params)
                logger.info(f"num params: {len(params)}")
                return state
    """
    def training_epoch_start(self, state: TrainState) -> TrainState:
        """
        Called at the beginning of every training epoch

        Parameters
        ----------
        state: TrainState
            The training state

        Returns
        -------
        TrainState
            The (possibly updated) training state
        """
        return state

    def training_epoch_end(self, state: TrainState) -> TrainState:
        """
        Called at the end of every training epoch

        Parameters
        ----------
        state: TrainState
            The training state

        Returns
        -------
        TrainState
            The (possibly updated) training state
        """
        return state

    def training_step_end(self, state: TrainState) -> TrainState:
        """
        Called after every minibatch update

        Parameters
        ----------
        state: TrainState
            The training state

        Returns
        -------
        TrainState
            The (possibly updated) training state
        """
        return state
