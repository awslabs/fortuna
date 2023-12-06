import abc

import numpy as np


class BaseBinaryClassificationTemperatureScaling(abc.ABC):
    def __init__(self):
        self._temperature = None

    @abc.abstractmethod
    def fit(self, probs: np.ndarray, targets: np.ndarray, **kwargs):
        pass

    def predict_proba(self, probs: np.ndarray):
        return np.clip(probs / self._temperature, 0.0, 1.0)

    def predict(self, probs: np.ndarray):
        return (self.predict_proba(probs) >= 0.5).astype(int)

    @property
    def temperature(self):
        return self._temperature
