import pathlib
from typing import (
    Optional,
    Tuple,
    Type,
)
from flax.traverse_util import path_aware_map, flatten_dict
from jax import (
    pure_callback,
    random,
)
from flax.core import FrozenDict
from jax._src.prng import PRNGKeyArray
import orbax
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

    def load_state(
            self,
            checkpoint_dir: Path,
            keep_top_n_checkpoints: int = 2,
            checkpoint_type: str = "last"
    ) -> None:
        path = pathlib.Path(checkpoint_dir)
        all_params_path = path / "all/0"
        all_params_checkpointer = orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())
        self.state = SGMCMCPosteriorStateRepository(
            size=self.posterior_approximator.n_samples,
            partition_manager=self.partition_manager,
            checkpoint_manager=get_checkpoint_manager(checkpoint_dir=str(path / "chain")),
            checkpoint_type=None,
        )
        if all_params_path.exists():
            self.state._which_params = tuple([list(p) for p in flatten_dict(path_aware_map(lambda p, v: p, self.state.get(0).params)).keys()])
            self.state._all_params = FrozenDict(all_params_checkpointer.restore(path / "all/0/default"))

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
                    checkpoint_dir=str(pathlib.Path(checkpoint_manager.directory) / str(checkpoint_manager.latest_step())),
                    keep_top_n_checkpoints=checkpoint_manager._options.max_to_keep
                ),
            )
            state = repo.get(optimizer=fit_config.optimizer.method, _do_reshard=_do_reshard)

            if fit_config.checkpointer.restore_checkpoint_dir is not None and fit_config.optimizer.freeze_fun is not None:
                all_params_checkpointer = orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())
                all_params_path = pathlib.Path(fit_config.checkpointer.restore_checkpoint_dir) / "all/0/default"
                if all_params_path.exists():
                    state = state.replace(params=FrozenDict(all_params_checkpointer.restore(all_params_path)))

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
