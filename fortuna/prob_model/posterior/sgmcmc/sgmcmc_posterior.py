from typing import Optional, Tuple, Type
import pathlib

from jax._src.prng import PRNGKeyArray
from jax import pure_callback, random

from fortuna.prob_model.posterior.base import Posterior
from fortuna.prob_model.posterior.state import PosteriorState
from fortuna.prob_model.fit_config.base import FitConfig
from fortuna.prob_model.joint.state import JointState
from fortuna.prob_model.posterior.map.map_state import MAPState
from fortuna.prob_model.posterior.posterior_multi_state_repository import \
    PosteriorMultiStateRepository
from fortuna.typing import Path


class SGMCMCPosterior(Posterior):
    """Base SGMCMC posterior approximators class."""
    def sample(
        self,
        rng: Optional[PRNGKeyArray] = None,
        **kwargs,
    ) -> JointState:
        """
        Sample from the posterior distribution.

        Parameters
        ----------
        rng : Optional[PRNGKeyArray]
            A random number generator. If not passed, this will be taken from the attributes of this class.

        Returns
        -------
        JointState
            A sample from the posterior distribution.
        """
        if rng is None:
            rng = self.rng.get()
        state = pure_callback(
            lambda j: self.state.get(i=j),
            self.state.get(i=0),
            random.choice(rng, self.posterior_approximator.n_samples),
        )
        return JointState(
            params=state.params,
            mutable=state.mutable,
            calib_params=state.calib_params,
            calib_mutable=state.calib_mutable,
        )

    def load_state(self, checkpoint_dir: Path) -> None:
        try:
            self.restore_checkpoint(pathlib.Path(checkpoint_dir) / "0")
        except ValueError:
            raise ValueError(
                f"No checkpoint was found in `checkpoint_dir={checkpoint_dir}`."
            )
        self.state = PosteriorMultiStateRepository(
            size=self.posterior_approximator.n_samples,
            checkpoint_dir=checkpoint_dir,
        )

    def save_state(
        self, checkpoint_dir: Path, keep_top_n_checkpoints: int = 1
    ) -> None:
        for i in range(self.posterior_approximator.n_samples):
            self.state.put(
                state=self.state.get(i), i=i, keep=keep_top_n_checkpoints
            )

    def _restore_state_from_somewhere(
        self,
        fit_config: FitConfig,
        allowed_states: Optional[Tuple[Type[MAPState], ...]] = None,
    ) -> MAPState:
        if fit_config.checkpointer.restore_checkpoint_path is not None:
            restore_checkpoint_path = pathlib.Path(fit_config.checkpointer.restore_checkpoint_path) / "c"
            state = self.restore_checkpoint(
                restore_checkpoint_path=restore_checkpoint_path,
                optimizer=fit_config.optimizer.method,
            )
        elif fit_config.checkpointer.start_from_current_state is not None:
            state = self.state.get(i=self.state.size - 1,
                                   optimizer=fit_config.optimizer.method,
            )

        if allowed_states is not None and not isinstance(state, allowed_states):
            raise ValueError(f"The type of the restored checkpoint must be within {allowed_states}. "
                             f"However, {fit_config.checkpointer.restore_checkpoint_path} pointed to a state "
                             f"with type {type(state)}.")

        self._check_state(state)
        return state
