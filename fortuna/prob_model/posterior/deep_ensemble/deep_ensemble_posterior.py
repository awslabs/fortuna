from __future__ import annotations

import logging
import pathlib
from typing import (
    List,
    Optional,
    Tuple,
    Type,
)
from fortuna.utils.checkpoint import get_checkpoint_manager
from flax.core import FrozenDict
from jax import (
    pure_callback,
    random,
)
from copy import deepcopy
from orbax.checkpoint import CheckpointManager
from jax._src.prng import PRNGKeyArray
from fortuna.data.loader import DataLoader
from fortuna.prob_model.posterior.map.map_posterior import MAPPosterior
from fortuna.prob_model.posterior.map.map_approximator import MAPPosteriorApproximator
from fortuna.prob_model.fit_config.base import FitConfig
from fortuna.prob_model.joint.base import Joint
from fortuna.prob_model.joint.state import JointState
from fortuna.prob_model.posterior.base import Posterior
from fortuna.prob_model.posterior.deep_ensemble import DEEP_ENSEMBLE_NAME
from fortuna.prob_model.posterior.deep_ensemble.deep_ensemble_approximator import (
    DeepEnsemblePosteriorApproximator,
)
from fortuna.prob_model.posterior.state import PosteriorState
from fortuna.prob_model.posterior.map.map_posterior import MAPState
from fortuna.prob_model.posterior.map.map_trainer import ShardedMAPTrainer, MAPTrainer
from fortuna.prob_model.posterior.posterior_multi_state_repository import (
    PosteriorMultiStateRepository,
)
from fortuna.prob_model.posterior.posterior_state_repository import PosteriorStateRepository
from fortuna.prob_model.posterior.run_preliminary_map import run_preliminary_map
from fortuna.typing import (
    Path,
    Status,
)
from fortuna.utils.builtins import get_dynamic_scale_instance_from_model_dtype
from fortuna.utils.freeze import get_trainable_paths
from fortuna.utils.nested_dicts import (
    nested_get,
    nested_set,
)
from fortuna.partitioner.partition_manager.base import PartitionManager


logger = logging.getLogger(__name__)


class DeepEnsemblePosterior(Posterior):
    def __init__(
        self,
        joint: Joint,
        posterior_approximator: DeepEnsemblePosteriorApproximator,
        partition_manager: PartitionManager
    ):
        """
        Deep ensemble approximate posterior class.

        Parameters
        ----------
        joint: Joint
            Joint distribution.
        posterior_approximator: DeepEnsemble
            Deep ensemble posterior approximator.
        """
        super().__init__(joint=joint, posterior_approximator=posterior_approximator, partition_manager=partition_manager)

    def __str__(self):
        return DEEP_ENSEMBLE_NAME

    def fit(
        self,
        train_data_loader: DataLoader,
        val_data_loader: Optional[DataLoader] = None,
        fit_config: FitConfig = FitConfig(),
        map_fit_config: Optional[FitConfig] = None,
        **kwargs,
    ) -> List[Status]:
        def _fun(i: int):
            fit_config_i = deepcopy(fit_config)
            fit_config_i.checkpointer.save_checkpoint_dir = str(pathlib.Path(fit_config.checkpointer.save_checkpoint_dir) / str(i)) if fit_config.checkpointer.save_checkpoint_dir else None
            fit_config_i.checkpointer.restore_checkpoint_dir = str(pathlib.Path(fit_config.checkpointer.restore_checkpoint_dir) / str(i)) if fit_config.checkpointer.restore_checkpoint_dir else None
            map_posterior = MAPPosterior(
                self.joint, posterior_approximator=MAPPosteriorApproximator(), partition_manager=self.partition_manager
            )
            map_posterior.rng = self.rng
            if self.state is not None:
                map_posterior.state = self.state.state[i]

            status = map_posterior.fit(
                rng=map_posterior.rng.get(),
                train_data_loader=train_data_loader,
                val_data_loader=val_data_loader,
                fit_config=fit_config_i,
                **kwargs,
            )
            return map_posterior.state, status

        if self.state is None:
            self.state = PosteriorMultiStateRepository(
                size=self.posterior_approximator.ensemble_size,
                partition_manager=self.partition_manager,
                checkpoint_manager=get_checkpoint_manager(
                    checkpoint_dir=str(
                        pathlib.Path(fit_config.checkpointer.save_checkpoint_dir)
                    ) if fit_config.checkpointer.save_checkpoint_dir is not None else None,
                    keep_top_n_checkpoints=fit_config.checkpointer.keep_top_n_checkpoints,
                ),
                checkpoint_type=fit_config.checkpointer.checkpoint_type
            )

        status = []
        for i in range(self.posterior_approximator.ensemble_size):
            logging.info(
                f"Run {i+1} out of {self.posterior_approximator.ensemble_size}."
            )
            self.state.state[i], _status = _fun(i)
            status.append(_status)
        logging.info("Fit completed.")
        return status

    def sample(self, rng: Optional[PRNGKeyArray] = None, **kwargs) -> JointState:
        if rng is None:
            rng = self.rng.get()
        state = pure_callback(
            lambda j: self.state.get(i=j),
            self.state.get(i=0),
            random.choice(rng, self.posterior_approximator.ensemble_size),
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
        self.state = PosteriorMultiStateRepository(
            size=self.posterior_approximator.ensemble_size,
            partition_manager=self.partition_manager,
            checkpoint_manager=get_checkpoint_manager(checkpoint_dir=checkpoint_dir),
            checkpoint_type=checkpoint_type
        )

    def save_state(self, checkpoint_dir: Path, keep_top_n_checkpoints: int = 1) -> None:
        if self.state is None:
            raise ValueError(
                """No state available. You must first either fit the posterior distribution, or load a
            saved checkpoint."""
            )
        for i in range(self.posterior_approximator.ensemble_size):
            self.state.put(state=self.state.get(i), i=i, checkpoint_dir=checkpoint_dir, keep=keep_top_n_checkpoints)

    def _init_map_state(
        self, state: Optional[MAPState], data_loader: DataLoader, fit_config: FitConfig
    ) -> MAPState:
        if state is None or fit_config.optimizer.freeze_fun is None:
            state = super()._init_joint_state(data_loader)

            return MAPState.init(
                params=state.params,
                mutable=state.mutable,
                optimizer=fit_config.optimizer.method,
                calib_params=state.calib_params,
                calib_mutable=state.calib_mutable,
                dynamic_scale=get_dynamic_scale_instance_from_model_dtype(
                    getattr(self.joint.likelihood.model_manager.model, "dtype")
                    if hasattr(self.joint.likelihood.model_manager.model, "dtype")
                    else None
                ),
            )
        else:
            random_state = super()._init_joint_state(data_loader)
            trainable_paths = get_trainable_paths(
                state.params, fit_config.optimizer.freeze_fun
            )
            state = state.replace(
                params=FrozenDict(
                    nested_set(
                        d=state.params.unfreeze(),
                        key_paths=trainable_paths,
                        objs=tuple(
                            [
                                nested_get(d=random_state.params, keys=path)
                                for path in trainable_paths
                            ]
                        ),
                    )
                )
            )

        return state

    def _restore_state_from_somewhere(
        self,
        i: int,
        fit_config: FitConfig,
        allowed_states: Optional[Tuple[Type[PosteriorState], ...]] = None,
        partition_manager: Optional[PartitionManager] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
    ) -> MAPState:
        if checkpoint_manager is not None:
            repo = PosteriorStateRepository(
                partition_manager=partition_manager,
                checkpoint_manager=get_checkpoint_manager(
                    checkpoint_dir=str(pathlib.Path(getattr(checkpoint_manager, "directory")) / fit_config.checkpointer.checkpoint_type / str(i)),
                    keep_top_n_checkpoints=checkpoint_manager._options.max_to_keep if checkpoint_manager is not None else None
                ),
            )
            state = repo.get(optimizer=fit_config.optimizer.method)
        elif fit_config.checkpointer.start_from_current_state:
            state = self.state.get(i=i, optimizer=fit_config.optimizer.method)

        if allowed_states is not None and not isinstance(state, allowed_states):
            raise ValueError(
                f"The type of the restored checkpoint must be within {allowed_states}. "
                f"However, the restored checkpoint has type {type(state)}."
            )

        return state
