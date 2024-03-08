from typing import (
    Callable,
    Optional,
)

import flax.linen as nn
import jax.numpy as jnp
import numpy as np

from fortuna.loss.classification.focal_loss import focal_loss_fn
from fortuna.output_calib_model.base import OutputCalibModel
from fortuna.output_calib_model.config.base import Config
from fortuna.output_calib_model.predictive.classification import (
    ClassificationPredictive,
)
from fortuna.output_calibrator.classification import ClassificationTemperatureScaler
from fortuna.output_calibrator.output_calib_manager.base import OutputCalibManager
from fortuna.prob_output_layer.classification import ClassificationProbOutputLayer
from fortuna.typing import (
    Array,
    Outputs,
    Status,
    Targets,
)


class OutputCalibClassifier(OutputCalibModel):
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
        self,
        calib_outputs: Array,
        calib_targets: Array,
        val_outputs: Optional[Array] = None,
        val_targets: Optional[Array] = None,
        loss_fn: Callable[[Outputs, Targets], jnp.ndarray] = focal_loss_fn,
        config: Config = Config(),
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
        loss_fn: Callable[[Outputs, Targets], jnp.ndarray]
            The loss function to use for calibration.
        config : Config
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
            uncertainty_fn=(
                config.monitor.uncertainty_fn
                if config.monitor.uncertainty_fn is not None
                else self.prob_output_layer.mean
            ),
            calib_outputs=calib_outputs,
            calib_targets=calib_targets,
            val_outputs=val_outputs,
            val_targets=val_targets,
            loss_fn=loss_fn,
            config=config,
        )

    @staticmethod
    def _check_output_dim(outputs: jnp.ndarray, targets: jnp.array):
        n_classes = len(np.unique(targets))
        if outputs.shape[1] != n_classes:
            raise ValueError(
                f"""`outputs.shape[1]` must be the same as the dimension of the number of classes in `targets`.
                However, `outputs.shape[1]={outputs.shape[1]}` and `len(np.unique(targets))={n_classes}`."""
            )
