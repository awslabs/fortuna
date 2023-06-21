import abc
import logging
from typing import (
    Any,
    Dict,
    Optional,
    Tuple,
    Type,
)

from flax.core import FrozenDict
from jax._src.prng import PRNGKeyArray

from fortuna.data.loader import DataLoader
from fortuna.prob_model.fit_config.base import FitConfig
from fortuna.prob_model.joint.base import Joint
from fortuna.prob_model.joint.state import JointState
from fortuna.prob_model.posterior.posterior_mixin import WithPosteriorCheckpointingMixin
from fortuna.prob_model.posterior.posterior_state_repository import (
    PosteriorStateRepository,
)
from fortuna.prob_model.posterior.state import PosteriorState
from fortuna.typing import (
    Path,
    Status,
)
from fortuna.utils.freeze import get_trainable_paths
from fortuna.utils.nested_dicts import (
    nested_get,
    nested_set,
)
from fortuna.utils.random import WithRNG


class PosteriorApproximator(abc.ABC):
    """
    A posterior approximator abstract class.
    """

    @abc.abstractmethod
    def __str__(self):
        pass

    @property
    def posterior_method_kwargs(self) -> Dict[str, Any]:
        return {}


class Posterior(WithRNG, WithPosteriorCheckpointingMixin):
    state = None

    def __init__(self, joint: Joint, posterior_approximator: PosteriorApproximator):
        r"""
        Posterior distribution class. This refers to :math:`p(w|\mathcal{D}, \phi)`, where :math:`w` are the random
        model parameters, :math:`\mathcal{D}` is a training data set and :math:`\phi` are calibration parameters.

        Parameters
        ----------
        joint: Joint
            A joint distribution object.
        posterior_approximator: PosteriorApproximator
            A posterior approximator.
        """
        super().__init__()
        self.joint = joint
        self.posterior_approximator = posterior_approximator

    def _restore_state_from_somewhere(
        self,
        fit_config: FitConfig,
        allowed_states: Optional[Tuple[Type[PosteriorState], ...]] = None,
    ) -> PosteriorState:
        if fit_config.checkpointer.restore_checkpoint_path is not None:
            state = self.restore_checkpoint(
                restore_checkpoint_path=fit_config.checkpointer.restore_checkpoint_path,
                optimizer=fit_config.optimizer.method,
            )
        elif fit_config.checkpointer.start_from_current_state is not None:
            state = self.state.get(optimizer=fit_config.optimizer.method)

        if allowed_states is not None and not isinstance(state, allowed_states):
            raise ValueError(
                f"The type of the restored checkpoint must be within {allowed_states}. "
                f"However, {fit_config.checkpointer.restore_checkpoint_path} pointed to a state "
                f"with type {type(state)}."
            )

        return state

    def _init_joint_state(self, data_loader: DataLoader) -> JointState:
        return self.joint.init(input_shape=data_loader.input_shape)

    @staticmethod
    def _freeze_optimizer_in_state(
        state: PosteriorState, fit_config: FitConfig
    ) -> PosteriorState:
        if fit_config.optimizer.freeze_fun is not None:
            trainable_paths = get_trainable_paths(
                state.params, fit_config.optimizer.freeze_fun
            )
            state = state.replace(
                opt_state=fit_config.optimizer.method.init(
                    FrozenDict(
                        nested_set(
                            d={},
                            key_paths=trainable_paths,
                            objs=tuple(
                                [
                                    nested_get(state.params.unfreeze(), path)
                                    for path in trainable_paths
                                ]
                            ),
                            allow_nonexistent=True,
                        )
                    )
                )
            )
        return state

    @staticmethod
    def _check_state(state: PosteriorState) -> None:
        pass

    @abc.abstractmethod
    def fit(
        self,
        train_data_loader: DataLoader,
        val_data_loader: Optional[DataLoader] = None,
        fit_config: FitConfig = FitConfig(),
        **kwargs,
    ) -> Status:
        """
        Fit the posterior distribution. A posterior state will be internally stored.

        Parameters
        ----------
        train_data_loader: DataLoader
            Training data loader.
        val_data_loader: Optional[DataLoader]
            Validation data loader.
        fit_config: FitConfig
            A configuration object.

        Returns
        -------
        Status
            A status including metrics describing the fitting process.
        """
        pass

    @abc.abstractmethod
    def sample(self, rng: Optional[PRNGKeyArray] = None, *args, **kwargs) -> JointState:
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
        pass

    def load_state(self, checkpoint_path: Path) -> None:
        """
        Load the state of the posterior distribution from a checkpoint path. The checkpoint must be
        compatible with the current probabilistic model.

        Parameters
        ----------
        checkpoint_path: Path
            Path to checkpoint file or directory to restore.
        """
        try:
            self.restore_checkpoint(checkpoint_path)
        except ValueError:
            raise ValueError(
                f"No checkpoint was found in `checkpoint_path={checkpoint_path}`."
            )
        self.state = PosteriorStateRepository(checkpoint_dir=checkpoint_path)

    def save_state(
        self, checkpoint_path: Path, keep_top_n_checkpoints: int = 1
    ) -> None:
        """
        Save the state of the posterior distribution to a checkpoint directory.

        Parameters
        ----------
        checkpoint_path: Path
            Path to checkpoint file or directory to restore.
        keep_top_n_checkpoints: int
            Number of past checkpoint files to keep.
        """
        if self.state is None:
            raise ValueError(
                """No state available. You must first either fit the posterior distribution, or load a
            saved checkpoint."""
            )
        return self.state.put(
            self.state.get(),
            checkpoint_path=checkpoint_path,
            keep=keep_top_n_checkpoints,
        )

    def _check_fit_config(self, fit_config: FitConfig):
        if (
            fit_config.checkpointer.dump_state is True
            and not fit_config.checkpointer.save_checkpoint_dir
        ):
            raise ValueError(
                "`save_checkpoint_dir` must be passed when `dump_state` is set to True."
            )

    @staticmethod
    def _is_state_available_somewhere(fit_config: FitConfig) -> bool:
        return (
            fit_config.checkpointer.restore_checkpoint_path is not None
            or fit_config.checkpointer.start_from_current_state
        )

    def _warn_frozen_params_start_from_random(
        self, fit_config: FitConfig, map_fit_config: Optional[FitConfig]
    ) -> None:
        if (
            not self._is_state_available_somewhere(fit_config)
            and map_fit_config is None
            and fit_config.optimizer.freeze_fun is not None
        ):
            logging.warning(
                "Parameters frozen via `fit_config.optimizer.freeze_fun` will not be updated. To start "
                "from sensible frozen parameters, you should configure "
                "`fit_config.checkpointer.restore_checkpoint_path`, or "
                "`fit_config.checkpointer.start_from_current_state`, or `map_fit_config`. "
                "Otherwise, "
                "a randomly initialized configuration of frozen parameters will be returned."
            )

    def _checks_on_fit_start(
        self, fit_config: FitConfig, map_fit_config: Optional[FitConfig]
    ) -> None:
        self._check_fit_config(fit_config)
        self._warn_frozen_params_start_from_random(fit_config, map_fit_config)

    def _should_run_preliminary_map(
        self, fit_config: FitConfig, map_fit_config: Optional[FitConfig]
    ) -> bool:
        return (
            not self._is_state_available_somewhere(fit_config)
            and map_fit_config is not None
        )
