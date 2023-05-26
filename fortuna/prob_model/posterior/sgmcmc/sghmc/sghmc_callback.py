from fortuna.prob_model.posterior.sgmcmc.sgmcmc_sampling_callback import (
    SGMCMCSamplingCallback,
)
from fortuna.training.callback import Callback
from fortuna.training.train_state import TrainState
from fortuna.training.train_state_repository import TrainStateRepository
from fortuna.training.trainer import TrainerABC


class SGHMCSamplingCallback(SGMCMCSamplingCallback):
    def __init__(
        self,
        n_epochs: int,
        n_training_steps: int,
        n_samples: int,
        n_thinning: int,
        burnin_length: int,
        trainer: TrainerABC,
        state_repository: TrainStateRepository,
        keep_top_n_checkpoints: int,
    ):
        """
        Stochastic Gradient Hamiltonian Monte Carlo (SGHMC) callback that collects samples
        after the initial burn-in phase.

        Parameters
        ----------
        n_epochs: int
            The number of epochs.
        n_training_steps: int
            The number of steps per epoch.
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
        """
        super().__init__(
            trainer=trainer,
            state_repository=state_repository,
            keep_top_n_checkpoints=keep_top_n_checkpoints,
        )

        self._do_sample = (
            lambda current_step, samples_count: samples_count < n_samples
            and current_step > burnin_length
            and (current_step - burnin_length) % n_thinning == 0
        )

        total_samples = sum(
            self._do_sample(step, 0)
            for step in range(1, n_epochs * n_training_steps + 1)
        )
        if total_samples < n_samples:
            raise ValueError(
                f"The number of desired samples `n_samples` is {n_samples}. However, only "
                f"{total_samples} samples will be collected. Consider adjusting the burnin "
                "length, number of epochs, or the thinning parameter."
            )
