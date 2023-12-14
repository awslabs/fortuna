import abc

import numpy as np
from scipy.optimize import brute
from scipy.special import (
    log_softmax,
    softmax,
)


class ClassificationTemperatureScaling(abc.ABC):
    def __init__(self):
        """
        A temperature scaling class for classification. It scales the logits with a shared learnable parameters.
        """
        self._temperature = None

    def fit(self, probs: np.ndarray, targets: np.ndarray):
        """
        Fit che temperature.

        Parameters
        ----------
        probs: np.ndarray
            A two-dimensional array of probabilities, for each input and class.
        targets: np.ndarray
            A one-dimensional array of integer target variables.
        """
        self._check_probs(probs)
        self._check_targets(targets)
        log_probs = np.log(probs)
        one_hot_targets = np.eye(probs.shape[1])[targets.reshape(-1)]

        def temp_scaling_fn(tau):
            log_temp_probs = log_softmax(log_probs / tau, axis=1)
            return -np.sum(one_hot_targets * log_temp_probs)

        self._temperature = brute(temp_scaling_fn, ranges=[(1e-6, 10)], Ns=1000)[0]

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        """
        Predict the scaled probabilities.

        Parameters
        ----------
        probs: np.ndarray
            A two-dimensional array of probabilities, for each input and class.

        Returns
        -------
        np.ndarray
            The predicted probabilities for each input and class.
        """
        self._check_probs(probs)
        return softmax(np.log(probs) / self._temperature, axis=1)

    def predict(self, probs: np.ndarray) -> np.ndarray:
        """
        Predict the target variable with the largest probability.

        Parameters
        ----------
        probs: np.ndarray
            A two-dimensional array of probabilities, for each input and class.

        Returns
        -------
        np.ndarray
            The predicted target variables for each input.
        """
        self._check_probs(probs)
        return np.argmax(probs, axis=1)

    @property
    def temperature(self):
        return self._temperature

    @staticmethod
    def _check_probs(probs: np.ndarray):
        if probs.ndim != 2:
            raise ValueError("The array of probabilities must be two-dimensional.")

    @staticmethod
    def _check_targets(targets: np.ndarray):
        if targets.ndim != 1:
            raise ValueError("The array of targets must be one-dimensional.")
        if targets.dtype != int:
            raise ValueError("Each element in the array of targets must be an integer.")
