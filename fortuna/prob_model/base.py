import abc
import logging
from typing import Dict, Optional

import jax.scipy.optimize as jspo
from fortuna.calib_config.base import CalibConfig
from fortuna.data.loader import DataLoader
from fortuna.output_calibrator.output_calib_manager.state import \
    OutputCalibManagerState
from fortuna.prob_model.fit_config import FitConfig
from fortuna.typing import Path, Status
from fortuna.utils.data import check_data_loader_is_not_random
from fortuna.utils.random import RandomNumberGenerator
from jax.flatten_util import ravel_pytree


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
        **fit_kwargs,
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
            A validation data loader.
        calib_data_loader : DataLoader
            A calibration data loader. If this is not passed, no calibration is performed.
        fit_config : FitConfig
            An object to configure the posterior distribution fitting.
        calib_config : CalibConfig
            An object to configure the calibration.
        fit_kwargs : dict
            Other arguments relevant to fitting the posterior distribution.

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
            **fit_kwargs,
        )
        logging.info("Fit completed.")

        calib_status = None
        if calib_data_loader:
            calib_status = self.calibrate(
                data_loader=calib_data_loader, calib_config=calib_config
            )
            logging.info("Calibration completed.")
        return dict(fit_status=fit_status, calib_status=calib_status)

    def calibrate(
        self, data_loader: DataLoader, calib_config: CalibConfig = CalibConfig(),
    ) -> Status:
        """
        Calibrate the probabilistic model.

        Parameters
        ----------
        data_loader : DataLoader
            A data loader.
        calib_config : CalibConfig
            An object to configure the calibration.

        Returns
        -------
        Status
            A calibration status object. It provides information about the calibration.
        """
        check_data_loader_is_not_random(data_loader)
        if self.output_calib_manager is None:
            logging.warning(
                """Nothing to calibrate. No calibrator was passed to the probabilistic model."""
            )
        else:
            if calib_config.optimizer.minimizer_kwargs is None:
                calib_config.optimizer.minimizer_kwargs = dict()

            if self.posterior.state is None:
                raise ValueError(
                    """Before calibration, you must either train the probabilistic model (see 
                `self.train`), or set a state from an existing checkpoint (see `self.set_state`)."""
                )

            ensemble_outputs = self.predictive._sample_calibrated_outputs(
                inputs_loader=data_loader.to_inputs_loader(),
                n_output_samples=calib_config.processor.n_posterior_samples,
            )

            if not calib_config.checkpointer.start_from_current_state:
                if calib_config.monitor.verbose:
                    logging.info("Initialize a calibration state.")
                ocms = OutputCalibManagerState.init_from_dict(
                    dict(
                        output_calibrator=self.output_calib_manager.init(
                            output_dim=ensemble_outputs.shape[-1]
                        )
                    )
                )
                calib_params = ocms.params
                calib_mutable = ocms.mutable
            else:
                if calib_config.monitor.verbose:
                    logging.info(
                        "Start from the current calibration state of the probabilistic model."
                    )
                d = self.posterior.state.extract_calib_keys()
                calib_params = d["calib_params"]
                calib_mutable = d["calib_mutable"]

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
                logp, aux = self.predictive.log_prob(
                    data_loader=data_loader,
                    n_posterior_samples=calib_config.processor.n_posterior_samples,
                    ensemble_outputs=ensemble_outputs,
                    calib_params=calib_params,
                    # calib_mutable=calib_mutable, # TODO: allow updating calibration mutable
                    return_aux=["calib_mutable"],
                )
                return -logp

            logging.info("Calibrate the probabilistic model...")
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
                k: getattr(result, k)
                for k in result._fields
                if k not in ["x", "status"]
            }

            self.posterior.state.update(
                dict(calib_params=calib_params, calib_mutable=calib_mutable),
                checkpoint_path=calib_config.checkpointer.save_state_path,
                keep=calib_config.checkpointer.keep_top_n_checkpoints,
            )
            return calib_status

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
