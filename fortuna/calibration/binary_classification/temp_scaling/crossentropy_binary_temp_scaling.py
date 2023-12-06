from typing import Dict

import numpy as np
from scipy.optimize import newton

from fortuna.calibration.binary_classification.temp_scaling.base import (
    BaseBinaryClassificationTemperatureScaling,
)


class CrossEntropyBinaryClassificationTemperatureScaling(
    BaseBinaryClassificationTemperatureScaling
):
    def fit(self, probs: np.ndarray, targets: np.ndarray, **kwargs) -> Dict:
        scaled_probs = (1 - 1e-6) * (1e-6 + probs)

        def temp_scaling_fn(phi):
            return np.mean((1 - targets) / (1 - scaled_probs * np.exp(-phi))) - 1

        phi, status = newton(temp_scaling_fn, x0=0.0, full_output=True, disp=False)
        self._temperature = np.exp(phi)
        return status
