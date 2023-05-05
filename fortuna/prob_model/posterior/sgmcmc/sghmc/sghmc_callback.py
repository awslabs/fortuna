import logging
from typing import Optional
import pathlib

from fortuna.training.train_state import TrainState
from fortuna.training.callback import Callback
from fortuna.training.train_state_repository import TrainStateRepository
from fortuna.training.trainer import TrainerABC
from fortuna.typing import Path

logger = logging.getLogger(__name__)


class SGHMCSamplingCallback(Callback):
    def __init__(self,
                 n_samples: int,
                 n_thinning: int,
                 burnin_length: int,
                 trainer: TrainerABC,
                 state_repository: TrainStateRepository,
                 keep_top_n_checkpoints: int,
                 save_checkpoint_dir: Optional[Path] = None,
                 ):
        """
        Stochastic Gradient Hamiltonian Monte Carlo (SGHMC) callback that collects samples
        after the initial burn-in phase.

        Parameters
        ----------
        n_samples: int
            The desired number of the posterior samples.
        n_thinning: int
            Keep only each `n_thinning` sample during the sampling phase.
        burnin_length: int
            Length of the initial burn-in phase, in steps.
        trainer: TrainerABC
            An instance of the trainer class.
        state_repository: TrainStateRepository
            An instance of the state repository.
        keep_top_n_checkpoints: int
            Number of past checkpoint files to keep.
        save_checkpoint_dir: Optional[Path]
            The optional path to save checkpoints.
        """
        self.n_samples = n_samples
        self.n_thinning = n_thinning
        self.burnin_length = burnin_length
        self.trainer = trainer
        self.state_repository = state_repository
        self.keep_top_n_checkpoints = keep_top_n_checkpoints
        self.save_checkpoint_dir = save_checkpoint_dir

        self.current_step = 0
        self.samples_count = 0

    def do_sample(self) -> bool:
        return self.current_step > self.burnin_length \
            and (self.current_step - self.burnin_length) % self.n_thinning == 0 \
            and self.samples_count < self.n_samples

    def training_step_end(self, state: TrainState) -> TrainState:
        self.current_step += 1

        if self.do_sample():
            if self.save_checkpoint_dir:
                self.trainer.save_checkpoint(
                    state,
                    pathlib.Path(self.save_checkpoint_dir)
                    / str(self.samples_count),
                    force_save=True,
                )
            self.state_repository.put(
                state=state,
                i=self.samples_count,
                keep=self.keep_top_n_checkpoints,
            )
            self.samples_count += 1

        return state
