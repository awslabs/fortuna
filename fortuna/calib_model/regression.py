from typing import Optional

import flax.linen as nn
import jax.numpy as jnp
from fortuna.calib_config.base import CalibConfig
from fortuna.calib_model.base import CalibModel
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
        self, outputs: Array, targets: Array, calib_config: CalibConfig = CalibConfig(),
    ) -> Status:
        self._check_output_dim(outputs, targets)
        return super().calibrate(outputs, targets, calib_config)

    def _check_output_dim(self, outputs: jnp.ndarray, targets: jnp.array):
        if outputs.shape[1] != 2 * targets.shape[1]:
            raise ValueError(
                f"""`outputs.shape[1]` must be twice the dimension of the target variables in `targets`, with 
                first and second halves corresponding to the mean and log-variance of the likelihood, respectively. 
                However, `outputs.shape[1]={outputs.shape[1]}` and `targets.shape[1]={targets.shape[1]}`."""
            )
