import abc
from typing import Optional, Tuple, Union, Dict, Any

from jax._src.prng import PRNGKeyArray

from fortuna.data.loader import DataLoader
from fortuna.prob_model.fit_config import FitConfig
from fortuna.prob_model.joint.base import Joint
from fortuna.prob_model.joint.state import JointState
from fortuna.prob_model.posterior.posterior_mixin import \
    WithPosteriorCheckpointingMixin
from fortuna.prob_model.posterior.posterior_state_repository import \
    PosteriorStateRepository
from fortuna.typing import Path, Status
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

    def _init(
        self,
        train_data_loader: DataLoader,
        val_data_loader: Optional[DataLoader] = None,
    ) -> Tuple[JointState, int, Union[int, None]]:
        n_train_data = 0
        for i, (batch_inputs, batch_targets) in enumerate(train_data_loader):
            n_train_data += batch_targets.shape[0]
            if i == 0:
                input_shape = batch_inputs.shape[1:]

        n_val_data = None
        if val_data_loader is not None:
            n_val_data = 0
            for batch in val_data_loader:
                n_val_data += batch[1].shape[0]

        return self.joint.init(input_shape), n_train_data, n_val_data

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
