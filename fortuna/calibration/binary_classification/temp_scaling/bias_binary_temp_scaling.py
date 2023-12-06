import numpy as np

from fortuna.calibration.binary_classification.temp_scaling.base import (
    BaseBinaryClassificationTemperatureScaling,
)


class BiasBinaryClassificationTemperatureScaling(
    BaseBinaryClassificationTemperatureScaling
):
    def fit(self, probs: np.ndarray, targets: np.ndarray, **kwargs):
        self._temperature = np.mean(probs) / np.mean(targets)
