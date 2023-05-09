from typing import Optional
import pathlib

from fortuna.training.train_state import TrainState
from fortuna.training.callback import Callback
from fortuna.training.train_state_repository import TrainStateRepository
from fortuna.training.trainer import TrainerABC
from fortuna.typing import Path


class SGMCMCSamplingCallback(Callback):
    def __init__(self,
                 trainer: TrainerABC,
                 state_repository: TrainStateRepository,
                 keep_top_n_checkpoints: int,
                 save_checkpoint_dir: Optional[Path] = None,
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
        save_checkpoint_dir: Optional[Path]
            The optional path to save checkpoints.
        """
        self._trainer = trainer
        self._state_repository = state_repository
        self._keep_top_n_checkpoints = keep_top_n_checkpoints
        self._save_checkpoint_dir = save_checkpoint_dir

        self._current_step = 0
        self._samples_count = 0

    def _do_sample(self, current_step, samples_count):
        raise NotImplementedError

    def training_step_end(self, state: TrainState) -> TrainState:
        self._current_step += 1

        if self._do_sample(self._current_step, self._samples_count):
            if self._save_checkpoint_dir:
                self._trainer.save_checkpoint(
                    state,
                    pathlib.Path(self._save_checkpoint_dir)
                    / str(self._samples_count),
                    force_save=True,
                )
            self._state_repository.put(
                state=state,
                i=self._samples_count,
                keep=self._keep_top_n_checkpoints,
            )
            self._samples_count += 1

        return state
