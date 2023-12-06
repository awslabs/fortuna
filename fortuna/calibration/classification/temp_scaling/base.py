import abc
from typing import Dict

import numpy as np
from scipy.optimize import minimize
from scipy.special import (
    log_softmax,
    softmax,
)


class ClassificationTemperatureScaling(abc.ABC):
    def __init__(self):
        self._temperature = None

    def fit(self, probs: np.ndarray, targets: np.ndarray, **kwargs) -> Dict:
        log_probs = np.log(probs)
        one_hot_targets = np.eye(probs.shape[1])[targets.reshape(-1)]
        n_data = probs.shape[0]

        def cross_entropy_fn(phi):
            log_temp_probs = log_softmax(log_probs * np.exp(-phi), axis=1)
            return -np.sum(one_hot_targets * log_temp_probs) / n_data

        res = minimize(
            cross_entropy_fn, np.array(0.0), options=dict(disp=False, **kwargs)
        )
        self._temperature = np.exp(float(res.x))
        return dict(message=res.message, success=res.success)

    def predict_proba(self, probs: np.ndarray):
        return softmax(np.log(probs) / self._temperature, axis=1)

    def predict(self, probs: np.ndarray):
        return np.argmax(self.predict_proba(probs), axis=1)

    @property
    def temperature(self):
        return self._temperature
