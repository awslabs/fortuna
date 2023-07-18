import pathlib
from typing import (
    Optional,
    Tuple,
    Type,
)

from jax import (
    pure_callback,
    random,
)
from jax._src.prng import PRNGKeyArray

from fortuna.prob_model.fit_config.base import FitConfig
from fortuna.prob_model.joint.state import JointState
from fortuna.prob_model.posterior.base import Posterior
from fortuna.prob_model.posterior.map.map_state import MAPState
from fortuna.prob_model.posterior.sgmcmc.sgmcmc_posterior_state_repository import (
    SGMCMCPosteriorStateRepository,
)
from fortuna.prob_model.posterior.posterior_state_repository import PosteriorStateRepository
from fortuna.utils.checkpoint import get_checkpoint_manager
from fortuna.partitioner.partition_manager.base import PartitionManager
from orbax.checkpoint import CheckpointManager
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
        self.state = SGMCMCPosteriorStateRepository(
            size=self.posterior_approximator.n_samples,
            partition_manager=self.partition_manager,
            checkpoint_manager=get_checkpoint_manager(checkpoint_dir=checkpoint_dir),
        )

    def save_state(self, checkpoint_dir: Path, keep_top_n_checkpoints: int = 1) -> None:
        if self.state is None:
            raise ValueError(
                """No state available. You must first either fit the posterior distribution, or load a
            saved checkpoint."""
            )
        for i in range(self.posterior_approximator.n_samples):
            self.state.put(state=self.state.get(i), i=i, checkpoint_dir=checkpoint_dir, keep=keep_top_n_checkpoints)

    def _restore_state_from_somewhere(
            self,
            fit_config: FitConfig,
            allowed_states: Optional[Tuple[Type[MAPState], ...]] = None,
            partition_manager: Optional[PartitionManager] = None,
            checkpoint_manager: Optional[CheckpointManager] = None,
            _do_reshard: bool = True
    ) -> MAPState:
        if checkpoint_manager is not None:
            repo = PosteriorStateRepository(
                partition_manager=None,
                checkpoint_manager=get_checkpoint_manager(
                    checkpoint_dir=str(pathlib.Path(checkpoint_manager.directory) / "c") if checkpoint_manager is not None else None,
                    keep_top_n_checkpoints=checkpoint_manager._options.max_to_keep if checkpoint_manager is not None else None
                ),
            )
            state = repo.get(optimizer=fit_config.optimizer.method, _do_reshard=_do_reshard)
        elif fit_config.checkpointer.start_from_current_state:
            state = self.state.get(
                i=self.state.size - 1,
                optimizer=fit_config.optimizer.method,
            )

        if allowed_states is not None and not isinstance(state, allowed_states):
            raise ValueError(
                f"The type of the restored checkpoint must be within {allowed_states}. "
                f"However, the restored checkpoint has type {type(state)}."
            )

        self._check_state(state)
        return state
