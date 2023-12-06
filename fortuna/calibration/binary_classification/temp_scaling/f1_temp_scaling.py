import numpy as np
from scipy.optimize import brute


class F1BinaryClassificationTemperatureScaling:
    def __init__(self):
        super().__init__()
        self._threshold = None
        self._temperature = None

    def fit(self, probs: np.ndarray, targets: np.ndarray, threshold: float):
        self._threshold = threshold
        n_pos_targets = np.sum(targets)

        def loss_fn(tau):
            temp_preds = probs >= threshold * tau
            n_pos_preds = np.sum(temp_preds)
            n_joint = np.sum(targets * temp_preds)
            prec = n_joint / n_pos_preds if n_pos_preds > 0 else 0.0
            rec = n_joint / n_pos_targets
            if prec + rec == 0.0:
                return 0.0
            return -2 * prec * rec / (prec + rec)

        self._temperature = brute(
            loss_fn, ranges=[(np.min(probs), 1 / threshold)], Ns=1000
        )[0]

    def predict_proba(self, probs: np.ndarray):
        return np.clip(probs / self._temperature, 0.0, 1.0)

    def predict(self, probs: np.ndarray):
        return (self.predict_proba(probs) >= self._threshold).astype(int)

    @property
    def threshold(self):
        return self._threshold

    @property
    def temperature(self):
        return self._temperature
