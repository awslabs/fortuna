from typing import Optional

import jax.numpy as jnp

from fortuna.calibration.output_calib_model.predictive.base import Predictive
from fortuna.output_calibrator.output_calib_manager.base import \
    OutputCalibManager
from fortuna.prob_output_layer.classification import \
    ClassificationProbOutputLayer
from fortuna.typing import Array


class ClassificationPredictive(Predictive):
    def __init__(
        self,
        output_calib_manager: OutputCalibManager,
        prob_output_layer: ClassificationProbOutputLayer,
    ):
        super().__init__(
            output_calib_manager=output_calib_manager,
            prob_output_layer=prob_output_layer,
        )

    def mean(self, outputs: Array, calibrated: bool = True, **kwargs) -> jnp.ndarray:
        """
        Estimate the mean of the one-hot encoded target variable given the output, with respect to the predictive
        distribution.

        Parameters
        ----------
        outputs : Array
            Model outputs.
        calibrated : bool
            Whether the outputs should be calibrated when computing this method. If `calibrated` is set to True, the
            model must have been calibrated beforehand.

        Returns
        -------
        jnp.ndarray
            The estimated mean for each output.
        """
        return super().mean(outputs, calibrated, **kwargs)

    def mode(self, outputs: Array, calibrated: bool = True, **kwargs) -> jnp.ndarray:
        """
        Estimate the mode of the one-hot encoded target variable given the output, with respect to the predictive
        distribution.

        Parameters
        ----------
        outputs : Array
            Model outputs.
        calibrated : bool
            Whether the outputs should be calibrated when computing this method. If `calibrated` is set to True, the
            model must have been calibrated beforehand.

        Returns
        -------
        jnp.ndarray
            The estimated mode for each output.
        """
        return super().mode(outputs, calibrated, **kwargs)

    def variance(
        self, outputs: jnp.ndarray, calibrated: bool = True, **kwargs
    ) -> jnp.ndarray:
        """
        Estimate the variance of the one-hot encoded target variable given the output, with respect to the predictive
        distribution.

        Parameters
        ----------
        outputs : jnp.ndarray
            Model outputs.
        calibrated : bool
            Whether the outputs should be calibrated when computing this method. If `calibrated` is set to True, the
            model must have been calibrated beforehand.

        Returns
        -------
        jnp.ndarray
            The estimated variance for each output.
        """
        return super().variance(outputs, calibrated, **kwargs)

    def std(
        self,
        outputs: jnp.ndarray,
        variances: Optional[jnp.ndarray] = None,
        calibrated: bool = True,
    ) -> jnp.ndarray:
        """
        Estimate the standard deviation of the one-hot encoded target variable given the output, with respect to the
        predictive distribution.

        Parameters
        ----------
        outputs : jnp.ndarray
            Model outputs.
        variances: Optional[jnp.ndarray]
            Variance for each output.
        calibrated : bool
            Whether the outputs should be calibrated when computing this method. If `calibrated` is set to True, the
            model must have been calibrated beforehand.

        Returns
        -------
        jnp.ndarray
            The estimated standard deviation for each output.
        """
        return super().std(outputs, variances, calibrated)
