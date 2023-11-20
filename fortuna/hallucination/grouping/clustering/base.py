from typing import (
    List,
    Union,
)

import numpy as np


class GroupingModel:
    """
    Grouping model based on clustering of embeddings.
    """

    def fit(
        self,
        embeddings: np.ndarray,
        clustering_models: List,
    ) -> None:
        """
        Fit the model.

        Parameters
        ----------
        embeddings: np.ndarray
            Embeddings of inputs.
        clustering_models: List
            A list of clustering models. Each clustering model must include the following method:

            - `fit`, to fit the model;
            - `bic`, to compute the BIC;
            - `predict_proba`, to get the predicted probabilities;
            - `predict`, to get the predictions.
            An example of valid clustering model is `sklearn.mixture.GaussianMixture`.
        """
        self._store_embeddings_stats(embeddings)
        embeddings = self._normalize(embeddings)

        self._fit_best_model(embeddings, models=clustering_models)

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """
        For each input, predict the probability of belonging to each cluster.

        Parameters
        ----------
        embeddings: np.ndarray
            Embeddings of inputs.

        Returns
        -------
        np.ndarray
            Predicted probabilities.
        """
        if self._clustering_model is None:
            raise ValueError("The `fit` method must be run first.")
        embeddings = self._normalize(embeddings)
        return self._clustering_model.predict_proba(embeddings)

    def soft_predict(
        self, embeddings: np.ndarray, quantile_proba_threshold: float = 0.8
    ) -> np.ndarray:
        """
        For each input, predict which clusters the inputs are most likely to belong to.

        Parameters
        ----------
        embeddings: np.ndarray
            Embeddings of inputs.
        quantile_proba_threshold: float
            The threshold at which to compute quantiles of the group scores.

        Returns
        -------
        np.ndarray
            An array of bools determining whether an input is predicted to belong to a cluster or not.
        """
        probs = self.predict_proba(embeddings=embeddings)
        quantiles = self.compute_quantile(probs, quantile_proba_threshold)
        return probs > quantiles[None]

    def hard_predict(
        self,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        For each input, predict the most likely cluster it belongs to.

        Parameters
        ----------
        embeddings: np.ndarray
            Embeddings of inputs.

        Returns
        -------
        np.ndarray
            An array of bools determining whether an input is predicted to belong to a cluster or not.
            Exactly one True will be given for each input.
        """
        probs = self.predict_proba(embeddings=embeddings)

        bool_preds = np.zeros_like(probs, dtype=bool)
        bool_preds[np.arange(len(probs)), np.argmax(probs, axis=1)] = True
        return bool_preds

    @property
    def clustering_model(self):
        return self._clustering_model

    @staticmethod
    def compute_quantile(probs, threshold: float) -> Union[float, np.ndarray]:
        return np.quantile(probs, threshold, axis=0)

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        if self._mean is None or self._std is None:
            raise ValueError("The `fit` method must be run first.")
        embeddings = np.copy(embeddings)
        embeddings -= self._mean
        embeddings /= self._std
        return embeddings

    def _store_embeddings_stats(self, embeddings):
        self._mean = np.mean(embeddings, axis=0, keepdims=True)
        self._std = np.std(embeddings, axis=0, keepdims=True)

    def _fit_best_model(self, embeddings: np.ndarray, models: List) -> None:
        best_bic = np.inf
        for model in models:
            model = model.fit(embeddings)
            bic = model.bic(embeddings)
            if bic < best_bic:
                best_bic = bic
                best_model = model
        self._clustering_model = best_model
