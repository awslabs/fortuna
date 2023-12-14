import abc

import numpy as np


class BaseBinaryClassificationTemperatureScaling(abc.ABC):
    def __init__(self):
        self._temperature = None

    @abc.abstractmethod
    def fit(self, probs: np.ndarray, targets: np.ndarray):
        """
        Fit the temperature scaling method.

        Parameters
        ----------
        probs: np.ndarray
            A one-dimensional probabilities of positive target variables for each input.
        targets: np.ndarray
            A one-dimensional array of integer target variables for each input.
        """
        pass

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        """
        Predict the scaled probabilities for each input.

        Parameters
        ----------
        probs: np.ndarray
            A one-dimensional probabilities of positive target variables for each input.
        Returns
        -------
        np.ndarray
            The predicted probabilities
        """
        self._check_probs(probs)
        return np.clip(probs / self._temperature, 0.0, 1.0)

    def predict(self, probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict the target variable for each input.

        Parameters
        ----------
        probs: np.ndarray
            A one-dimensional probabilities of positive target variables for each input.
        threshold: np.ndarray
            The threshold on the predicted probabilities do decide whether a target variable is positive or
            negative.

        Returns
        -------
        np.ndarray
            The predicted target variables.
        """
        self._check_probs(probs)
        return (self.predict_proba(probs) >= threshold).astype(int)

    @property
    def temperature(self):
        return self._temperature

    @staticmethod
    def _check_probs(probs: np.ndarray):
        if probs.ndim != 1:
            raise ValueError("The array of probabilities must be one-dimensional.")

    @staticmethod
    def _check_targets(targets: np.ndarray):
        if targets.ndim != 1:
            raise ValueError("The array of targets must be one-dimensional.")
        if targets.dtype != int:
            raise ValueError("Each element in the array of targets must be an integer.")
