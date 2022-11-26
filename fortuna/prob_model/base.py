import abc
import logging
from typing import Callable, Dict, Optional

import jax
import jax.numpy as jnp

from fortuna.calibration.state import CalibState
from fortuna.data.loader import DataLoader
from fortuna.prob_model.calib_config.base import CalibConfig
from fortuna.prob_model.fit_config import FitConfig
from fortuna.prob_model.prob_model_calibrator import (
    JittedProbModelCalibrator, MultiGPUProbModelCalibrator,
    ProbModelCalibrator)
from fortuna.typing import Array, Path, Status
from fortuna.utils.data import check_data_loader_is_not_random
from fortuna.utils.gpu import select_trainer_given_devices
from fortuna.utils.random import RandomNumberGenerator


class ProbModel(abc.ABC):
    """
    Abstract probabilistic model class.
    """

    def __init__(self, seed: int = 0):
        self.rng = RandomNumberGenerator(seed=seed)
        self.__set_rng()

    def __set_rng(self):
        self.model_manager.rng = self.rng
        self.output_calib_manager.rng = self.rng
        self.prob_output_layer.rng = self.rng
        self.prior.rng = self.rng
        self.likelihood.rng = self.rng
        self.joint.rng = self.rng
        self.posterior.rng = self.rng
        self.predictive.rng = self.rng

    def train(
        self,
        train_data_loader: DataLoader,
        val_data_loader: Optional[DataLoader] = None,
        calib_data_loader: Optional[DataLoader] = None,
        fit_config: FitConfig = FitConfig(),
        calib_config: CalibConfig = CalibConfig(),
        map_fit_config: Optional[FitConfig] = None,
    ) -> Dict[str, Status]:
        """
        Train the probabilistic model. This involves fitting the posterior distribution and calibrating the
        probabilistic model. Calibration is performed only if (1) `calib_data_loader` is passed and (2) the
        probabilistic model contains any calibrator.

        Parameters
        ----------
        train_data_loader : DataLoader
            A training data loader.
        val_data_loader : DataLoader
            A validation data loader. This is used to validate both posterior fitting and calibration.
        calib_data_loader : DataLoader
            A calibration data loader. If this is not passed, no calibration is performed.
        fit_config : FitConfig
            An object to configure the posterior distribution fitting.
        calib_config : CalibConfig
            An object to configure the calibration.
        map_fit_config : Optional[FitConfig] = None
            An object to configure a preliminary posterior distribution fitting via the Maximum-A-Posteriori (MAP)
            method.
            The fit of several supported posterior approximation methods,
            like :class:`~fortuna.prob_model.posterior.swag.swag_posterior.SWAGPosterior.fit` and
            :class:`~fortuna.prob_model.posterior.swag.swag_posterior.LaplacePosterior.fit`, start from a preliminary
            run of MAP, which can be configured via this object. If the method does not start from MAP, this argument is
            ignored.

        Returns
        -------
        Dict[str, Status]
            Status objects for both posterior fitting and calibration.

        """
        logging.info("Fit the posterior distribution...")
        fit_status = self.posterior.fit(
            train_data_loader=train_data_loader,
            val_data_loader=val_data_loader,
            fit_config=fit_config,
            map_fit_config=map_fit_config,
        )
        logging.info("Fit completed.")

        calib_status = None
        if calib_data_loader:
            calib_status = self.calibrate(
                calib_data_loader=calib_data_loader,
                val_data_loader=val_data_loader,
                calib_config=calib_config,
            )
            logging.info("Calibration completed.")
        return dict(fit_status=fit_status, calib_status=calib_status)

    def _calibrate(
        self,
        calib_data_loader: DataLoader,
        uncertainty_fn: Callable[[jnp.ndarray, jnp.ndarray, Array], jnp.ndarray],
        val_data_loader: Optional[DataLoader] = None,
        calib_config: CalibConfig = CalibConfig(),
    ) -> Status:
        check_data_loader_is_not_random(calib_data_loader)
        if val_data_loader is not None:
            check_data_loader_is_not_random(val_data_loader)
        if self.output_calib_manager is None:
            logging.warning(
                """Nothing to calibrate. No calibrator was passed to the probabilistic model."""
            )
        else:
            if self.posterior.state is None:
                raise ValueError(
                    """Before calibration, you must either train the probabilistic model (see 
                        :meth:`~fortuna.prob_model.base.ProbModel.train`), 
                        or load a state from an existing checkpoint 
                        (see :meth:`~fortuna.prob_model.base.ProbModel.load_state`)."""
                )
            if calib_config.monitor.verbose:
                logging.info(
                    "Pre-compute ensemble of outputs on the calibration data loader."
                )

            distribute = jax.local_device_count() > 1

            (
                calib_ensemble_outputs_loader,
                calib_size,
            ) = self.predictive._sample_outputs_loader(
                inputs_loader=calib_data_loader.to_inputs_loader(),
                n_output_samples=calib_config.processor.n_posterior_samples,
                return_size=True,
                distribute=distribute
            )
            if calib_config.monitor.verbose:
                logging.info(
                    "Pre-compute ensemble of outputs on the validation data loader."
                )
            val_ensemble_outputs_loader, val_size = (
                self.predictive._sample_outputs_loader(
                    inputs_loader=val_data_loader.to_inputs_loader(),
                    n_output_samples=calib_config.processor.n_posterior_samples,
                    return_size=True,
                    distribute=distribute
                )
                if val_data_loader is not None
                else (None, None)
            )

            trainer_cls = select_trainer_given_devices(
                gpus=calib_config.processor.gpus,
                BaseTrainer=ProbModelCalibrator,
                JittedTrainer=JittedProbModelCalibrator,
                MultiGPUTrainer=MultiGPUProbModelCalibrator,
                disable_jit=calib_config.processor.disable_jit,
            )

            calibrator = trainer_cls(
                calib_outputs_loader=calib_ensemble_outputs_loader,
                val_outputs_loader=val_ensemble_outputs_loader,
                predict_fn=self.prob_output_layer.predict,
                uncertainty_fn=uncertainty_fn,
                save_checkpoint_dir=calib_config.checkpointer.save_checkpoint_dir,
                save_every_n_steps=calib_config.checkpointer.save_every_n_steps,
                keep_top_n_checkpoints=calib_config.checkpointer.keep_top_n_checkpoints,
                disable_training_metrics_computation=calib_config.monitor.disable_calibration_metrics_computation,
                eval_every_n_epochs=calib_config.monitor.eval_every_n_epochs,
            )

            if calib_config.checkpointer.restore_checkpoint_path is None:
                calib_dict = self.posterior.state.extract_calib_keys()

                state = CalibState.init(
                    params=calib_dict["calib_params"],
                    mutable=calib_dict["calib_mutable"],
                    optimizer=calib_config.optimizer.method,
                )
            else:
                state = self.posterior.restore_checkpoint(
                    calib_config.checkpointer.restore_checkpoint_path,
                    optimizer=calib_config.optimizer.method,
                )

            if calib_config.monitor.verbose:
                logging.info("Start calibration.")
            state, status = calibrator.train(
                rng=self.rng.get(),
                state=state,
                fun=self.predictive._batched_log_prob,
                training_data_loader=calib_data_loader,
                training_dataset_size=calib_size,
                n_epochs=calib_config.optimizer.n_epochs,
                metrics=calib_config.monitor.metrics,
                val_data_loader=val_data_loader,
                val_dataset_size=val_size,
                verbose=calib_config.monitor.verbose,
            )

            self.posterior.state.update(
                variables=dict(calib_params=state.params, calib_mutable=state.mutable)
            )

            if (
                calib_config.checkpointer.dump_state
                and calib_config.checkpointer.save_checkpoint_dir is not None
            ):
                if calib_config.monitor.verbose:
                    logging.info("Dump state to disk.")
                self.save_state(
                    checkpoint_path=calib_config.checkpointer.save_checkpoint_dir
                )

            if calib_config.monitor.verbose:
                logging.info("Calibration completed.")

            return status

    def load_state(self, checkpoint_path: Path) -> None:
        """
        Load the state of the posterior distribution from a checkpoint path. The checkpoint must be compatible with the
        probabilistic model.

        Parameters
        ----------
        checkpoint_path : Path
            Path to a checkpoint file or directory to restore.
        """
        return self.posterior.load_state(checkpoint_path)

    def save_state(
        self, checkpoint_path: Path, keep_top_n_checkpoints: int = 1
    ) -> None:
        """
        Save the posterior distribution state as a checkpoint.

        Parameters
        ----------
        checkpoint_path : Path
            Path to file or directory where to save the current state.
        keep_top_n_checkpoints : int
            Number of past checkpoint files to keep.
        """
        return self.posterior.save_state(checkpoint_path, keep_top_n_checkpoints)
