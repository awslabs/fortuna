import abc
import logging

import jax.numpy as jnp
import jax.scipy.optimize as jspo
from flax.core import FrozenDict
from fortuna.calib_config.base import CalibConfig
from fortuna.output_calibrator.output_calib_manager.state import \
    OutputCalibManagerState
from fortuna.training.calib_state import CalibState
from fortuna.training.mixin import WithCheckpointingMixin
from fortuna.training.train_state_repository import TrainStateRepository
from fortuna.typing import Array, Path, Status
from fortuna.utils.random import RandomNumberGenerator
from jax.flatten_util import ravel_pytree


class CalibModel(WithCheckpointingMixin, abc.ABC):
    """
    Abstract calibration model class.
    """

    def __init__(self, seed: int = 0):
        super().__init__()
        self.rng = RandomNumberGenerator(seed=seed)
        self.__set_rng()

    def __set_rng(self):
        self.output_calib_manager.rng = self.rng
        self.prob_output_layer.rng = self.rng
        self.predictive.rng = self.rng

    def calibrate(
        self, outputs: Array, targets: Array, calib_config: CalibConfig = CalibConfig(),
    ) -> Status:
        """
        Calibrate the model outputs.

        Parameters
        ----------
        outputs: Array
            Model outputs.
        targets: Array
            Array of target variables.
        calib_config : CalibConfig
            An object to configure the calibration.

        Returns
        -------
        Status
            A calibration status object. It provides information about the calibration.
        """
        if (
            not calib_config.checkpointer.start_from_current_state
            and calib_config.checkpointer.restore_checkpoint_path is None
        ):
            calib_state = OutputCalibManagerState.init_from_dict(
                FrozenDict(
                    output_calibrator=self.output_calib_manager.init(
                        output_dim=outputs.shape[-1]
                    )
                )
            )
        elif calib_config.checkpointer.start_from_current_state:
            calib_state = self.predictive.state.get()
        else:
            calib_state = self.restore_checkpoint(
                restore_checkpoint_path=calib_config.checkpointer.restore_checkpoint_path
            )
        calib_params = calib_state.params
        calib_mutable = calib_state.mutable

        if calib_mutable is not None and any(
            [v is not None for v in calib_mutable.values()]
        ):
            raise NotImplementedError(
                f"This does not currently support calibration models that include mutable "
                f"objects. However, the following calibration mutable objects were found: "
                f"{calib_mutable}"
            )

        rav, unravel = ravel_pytree(calib_params)

        def loss_fn(rav_calib_params):
            calib_params = unravel(rav_calib_params)
            calib_outputs = self.output_calib_manager.apply(
                params=calib_params["output_calibrator"],
                outputs=outputs,
                # mutable=calib_mutable, # TODO: allow updating calibration mutable
                calib=True,
            )
            logp = jnp.sum(
                self.prob_output_layer.log_prob(outputs=calib_outputs, targets=targets,)
            )
            return -logp

        logging.info("Calibrate the model...")
        if calib_config.optimizer.minimizer_kwargs is None:
            calib_config.optimizer.minimizer_kwargs = dict()
        result = jspo.minimize(
            loss_fn, rav, method="BFGS", **calib_config.optimizer.minimizer_kwargs
        )
        logging.info("Calibration completed.")
        if result.success is False:
            logging.warning(
                """The calibration algorithm did not successfully converge. Please use the solution with caution."""
            )
        calib_params = unravel(result.x)
        calib_status = {
            k: getattr(result, k) for k in result._fields if k not in ["x", "status"]
        }

        self.predictive.state = TrainStateRepository(
            checkpoint_dir=calib_config.checkpointer.save_state_path
        )
        self.predictive.state.put(
            state=CalibState.init(params=calib_params, mutable=calib_mutable),
        )
        return calib_status

    def load_state(self, checkpoint_path: Path) -> None:
        """
        Load a calibration state from a checkpoint path.
        The checkpoint must be compatible with the calibration model.

        Parameters
        ----------
        checkpoint_path : Path
            Path to a checkpoint file or directory to restore.
        """
        try:
            self.restore_checkpoint(checkpoint_path)
        except ValueError:
            raise ValueError(
                f"No checkpoint was found in `checkpoint_path={checkpoint_path}`."
            )
        self.predictive.state = TrainStateRepository(checkpoint_dir=checkpoint_path)

    def save_state(
        self, checkpoint_path: Path, keep_top_n_checkpoints: int = 1
    ) -> None:
        """
        Save the calibration state as a checkpoint.

        Parameters
        ----------
        checkpoint_path : Path
            Path to file or directory where to save the current state.
        keep_top_n_checkpoints : int
            Number of past checkpoint files to keep.
        """
        if self.predictive.state is None:
            raise ValueError(
                """No state available. You must first either calibrate the model, or load a saved checkpoint."""
            )
        return self.predictive.state.put(
            self.predictive.state.get(),
            checkpoint_path=checkpoint_path,
            keep=keep_top_n_checkpoints,
        )
