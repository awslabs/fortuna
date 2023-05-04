import logging
from typing import Optional
import pathlib

from fortuna.training.train_state import TrainState
from fortuna.training.callback import Callback
from fortuna.training.train_state_repository import TrainStateRepository
from fortuna.training.trainer import TrainerABC
from fortuna.typing import Path

logger = logging.getLogger(__name__)


class CyclicalSGLDSamplingCallback(Callback):
    def __init__(self,
                 n_epochs: int,
                 n_samples: int,
                 n_thinning: int,
                 cycle_length: int,
                 exploration_ratio: float,
                 trainer: TrainerABC,
                 state_repository: TrainStateRepository,
                 keep_top_n_checkpoints: int,
                 save_checkpoint_dir: Optional[Path] = None,
                 ):
        """
        Cyclical Stochastic Gradient Langevin Dynamics (SGLD) callback that collects samples
        in different cycles. See `Zhang R. et al., 2020 <https://openreview.net/pdf?id=rkeS1RVtPS>`_
        for more details.

        Parameters
        ----------
        """
        if n_samples * cycle_length < cycle_length != 0:
            raise ValueError("The number of desired samples per cycle `n_samples` * `n_thinning` "
                             f"= {n_samples * cycle_length} is less than `cycle_length` = {cycle_length}.")
        self.n_samples = n_samples
        self.n_thinning = n_thinning
        self.cycle_length = cycle_length
        self.exploration_ratio = exploration_ratio
        self.trainer = trainer
        self.state_repository = state_repository
        self.keep_top_n_checkpoints = keep_top_n_checkpoints
        self.save_checkpoint_dir = save_checkpoint_dir

        self.samples_per_epoch = n_samples // n_epochs
        self.current_step = 0
        self.current_epoch = 0
        self.samples_count = 0

    def do_sample(self) -> bool:
        return ((self.current_step % self.cycle_length) / self.cycle_length) < self.exploration_ratio \
            and (self.current_step % self.cycle_length) % self.n_thinning == 0 \
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

    def training_epoch_end(self, state: TrainState) -> TrainState:
        self.current_epoch += 1

        if self.samples_count / self.current_epoch < self.samples_per_epoch:
            logging.warning("The number of sampled states is less than the expected number of samples "
                            f"per epoch {self.samples_per_epoch}. Consider adjusting the cycle "
                            "length or the thinning parameter.")
        return state
