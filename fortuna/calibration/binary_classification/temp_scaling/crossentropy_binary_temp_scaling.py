import numpy as np
from scipy.optimize import brute

from fortuna.calibration.binary_classification.temp_scaling.base import (
    BaseBinaryClassificationTemperatureScaling,
)


class CrossEntropyBinaryClassificationTemperatureScaling(
    BaseBinaryClassificationTemperatureScaling
):
    """
    A temperature scaling class for binary classification.
    It scales the probability that the target variables is positive with a single learnable parameters.
    The method minimizes the binary cross-entropy loss.
    """

    def fit(self, probs: np.ndarray, targets: np.ndarray):
        self._check_probs(probs)
        self._check_targets(targets)

        def temp_scaling_fn(tau):
            temp_probs = np.clip(probs / tau, 1e-9, 1 - 1e-9)
            return -np.mean(
                targets * np.log(temp_probs) + (1 - targets) * np.log(1 - temp_probs)
            )

        self._temperature = brute(
            temp_scaling_fn, ranges=[(np.min(probs), 10)], Ns=1000
        )[0]
