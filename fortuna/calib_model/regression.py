from typing import Optional

import flax.linen as nn
import jax.numpy as jnp

from fortuna.calib_model.base import CalibModel
from fortuna.calib_model.calib_config.base import CalibConfig
from fortuna.calib_model.predictive.regression import RegressionPredictive
from fortuna.output_calibrator.output_calib_manager.base import \
    OutputCalibManager
from fortuna.output_calibrator.regression import RegressionTemperatureScaler
from fortuna.prob_output_layer.regression import RegressionProbOutputLayer
from fortuna.typing import Array, Status


class CalibRegressor(CalibModel):
    def __init__(
        self,
        output_calibrator: Optional[nn.Module] = RegressionTemperatureScaler(),
        seed: int = 0,
    ) -> None:
        r"""
        A calibration regressor class.

        Parameters
        ----------
        output_calibrator : Optional[nn.Module]
            An output calibrator object. The default is temperature scaling for regression, which inflates the variance
            of the likelihood with a scalar temperature parameter. Given outputs :math:`o` of the model manager, the
            output calibrator is described by a function :math:`g(\phi, o)`, where `phi` are
            calibration parameters.
        seed: int
            A random seed.

        Attributes
        ----------
        output_calibrator : nn.Module
            See `output_calibrator` in `Parameters`.
        output_calib_manager : OutputCalibManager
            It manages the forward pass of the output calibrator.
        prob_output_layer : RegressionProbOutputLayer
            A probabilistic output payer.
            It characterizes the distribution of the target variables given the outputs.
        predictive : RegressionPredictive
            The predictive distribution.
        """
        self.output_calibrator = output_calibrator
        self.output_calib_manager = OutputCalibManager(
            output_calibrator=output_calibrator
        )
        self.prob_output_layer = RegressionProbOutputLayer()
        self.predictive = RegressionPredictive(
            output_calib_manager=self.output_calib_manager,
            prob_output_layer=self.prob_output_layer,
        )
        super().__init__(seed=seed)

    def calibrate(
        self,
        calib_outputs: Array,
        calib_targets: Array,
        val_outputs: Optional[Array] = None,
        val_targets: Optional[Array] = None,
        calib_config: CalibConfig = CalibConfig(),
    ) -> Status:
        """
        Calibrate the model outputs.

        Parameters
        ----------
        calib_outputs: Array
            Calibration model outputs.
        calib_targets: Array
            Calibration target variables.
        val_outputs: Optional[Array]
            Validation model outputs.
        val_targets: Optional[Array]
            Validation target variables.
        calib_config : CalibConfig
            An object to configure the calibration.

        Returns
        -------
        Status
            A calibration status object. It provides information about the calibration.
        """
        self._check_output_dim(calib_outputs, calib_targets)
        if val_outputs is not None:
            self._check_output_dim(val_outputs, val_targets)
        return super()._calibrate(
            uncertainty_fn=calib_config.monitor.uncertainty_fn
            if calib_config.monitor.uncertainty_fn is not None
            else self.prob_output_layer.variance,
            calib_outputs=calib_outputs,
            calib_targets=calib_targets,
            val_outputs=val_outputs,
            val_targets=val_targets,
            calib_config=calib_config,
        )

    def _check_output_dim(self, outputs: jnp.ndarray, targets: jnp.array):
        if outputs.shape[1] != 2 * targets.shape[1]:
            raise ValueError(
                f"""`outputs.shape[1]` must be twice the dimension of the target variables in `targets`, with 
                first and second halves corresponding to the mean and log-variance of the likelihood, respectively. 
                However, `outputs.shape[1]={outputs.shape[1]}` and `targets.shape[1]={targets.shape[1]}`."""
            )
