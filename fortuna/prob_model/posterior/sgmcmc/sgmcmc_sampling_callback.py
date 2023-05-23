from typing import Optional

from fortuna.training.train_state import TrainState
from fortuna.training.callback import Callback
from fortuna.training.train_state_repository import TrainStateRepository
from fortuna.training.trainer import TrainerABC


class SGMCMCSamplingCallback(Callback):
    def __init__(
        self,
        trainer: TrainerABC,
        state_repository: TrainStateRepository,
        keep_top_n_checkpoints: int,
    ):
        """
        Sampling callback that collects samples from the MCMC chain.

        Parameters
        ----------
        trainer: TrainerABC
            An instance of the trainer class.
        state_repository: TrainStateRepository
            An instance of the state repository.
        keep_top_n_checkpoints: int
            Number of past checkpoint files to keep.
        """
        self._trainer = trainer
        self._state_repository = state_repository
        self._keep_top_n_checkpoints = keep_top_n_checkpoints

        self._current_step = 0
        self._samples_count = 0

    def _do_sample(self, current_step, samples_count):
        raise NotImplementedError

    def training_step_end(self, state: TrainState) -> TrainState:
        self._current_step += 1

        if self._do_sample(self._current_step, self._samples_count):
            self._state_repository.put(
                state=self._trainer._sync_state(state),
                i=self._samples_count,
                keep=self._keep_top_n_checkpoints,
            )
            self._samples_count += 1

        return state
