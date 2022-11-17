from typing import Optional

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from fortuna.calib_config.base import CalibConfig
from fortuna.calib_model.base import CalibModel
from fortuna.calib_model.predictive.classification import \
    ClassificationPredictive
from fortuna.output_calibrator.classification import \
    ClassificationTemperatureScaler
from fortuna.output_calibrator.output_calib_manager.base import \
    OutputCalibManager
from fortuna.prob_output_layer.classification import \
    ClassificationProbOutputLayer
from fortuna.typing import Array, Status


class CalibClassifier(CalibModel):
    def __init__(
        self,
        output_calibrator: Optional[nn.Module] = ClassificationTemperatureScaler(),
        seed: int = 0,
    ) -> None:
        r"""
        A calibration classifier class.

        Parameters
        ----------
        output_calibrator : Optional[nn.Module]
            An output calibrator object. The default is temperature scaling for classification, which rescales the
            logits with a scalar temperature parameter. Given outputs :math:`o`,
            the output calibrator is described by a function :math:`g(\phi, o)`,
            where `phi` are calibration parameters.
        seed: int
            A random seed.

        Attributes
        ----------
        output_calibrator : nn.Module
            See `output_calibrator` in `Parameters`.
        output_calib_manager : OutputCalibManager
            It manages the forward pass of the output calibrator.
        prob_output_layer : ClassificationProbOutputLayer
            A probabilistic output payer.
            It characterizes the distribution of the target variables given the outputs.
        predictive : ClassificationPredictive
            The predictive distribution.
        """
        self.output_calibrator = output_calibrator
        self.output_calib_manager = OutputCalibManager(
            output_calibrator=output_calibrator
        )
        self.prob_output_layer = ClassificationProbOutputLayer()
        self.predictive = ClassificationPredictive(
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
        n_classes = len(np.unique(targets))
        if outputs.shape[1] != n_classes:
            raise ValueError(
                f"""`outputs.shape[1]` must be the same as the dimension of the number of classes in `targets`. 
                However, `outputs.shape[1]={outputs.shape[1]}` and `len(np.unique(targets))={n_classes}`."""
            )
