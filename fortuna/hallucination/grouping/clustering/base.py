from typing import (
    Iterable,
    List,
    Optional,
)

import numpy as np

from fortuna.hallucination.embedding import EmbeddingManager
from fortuna.typing import Array


class GroupingModel:
    """
    Grouping model based on clustering of embeddings.
    """

    def __init__(
        self, embedding_manager: EmbeddingManager, quantile_proba_threshold: float = 0.8
    ):
        self.embedding_manager = embedding_manager
        self._clustering_model = None
        self._embeddings_mean = None
        self._embeddings_std = None
        self._quantiles = None
        self.quantile_proba_threshold = quantile_proba_threshold

    def fit(
        self,
        inputs: Iterable,
        clustering_models: List,
        extra_embeddings: Optional[Array] = None,
    ) -> None:
        """
        Fit the model.

        Parameters
        ----------
        inputs: Iterable
            An iterable of inputs.
        clustering_models: List
            A list of clustering models. Each clustering model must include the following method:
        extra_embeddings: Optional[Array]
            An extra array of embeddings.

            - `fit`, to fit the model;
            - `bic`, to compute the BIC;
            - `predict_proba`, to get the predicted probabilities;
            - `predict`, to get the predictions.
            An example of valid clustering model is `sklearn.mixture.GaussianMixture`.
        """
        if not isinstance(clustering_models, list) or not len(clustering_models):
            raise ValueError("`clustering_models` must be a non-empty list.")
        embeddings = self._get_concat_embeddings(inputs, extra_embeddings)
        self._store_embeddings_stats(embeddings)
        embeddings = self._normalize(embeddings)

        self._fit_best_model(embeddings, models=clustering_models)

        probs = self._clustering_model.predict_proba(embeddings)
        self._store_thresholds(
            probs=probs, quantile_threshold=self.quantile_proba_threshold
        )

    def predict_proba(
        self, inputs: Iterable, extra_embeddings: Optional[Array] = None
    ) -> Array:
        """
        For each input, predict the probability of belonging to each cluster.

        Parameters
        ----------
        inputs: Iterable
            An iterable of inputs.
        extra_embeddings: Optional[Array]
            An extra array of embeddings.

        Returns
        -------
        Array
            Predicted probabilities.
        """
        if self._clustering_model is None:
            raise ValueError("The `fit` method must be run first.")
        embeddings = self._get_concat_embeddings(inputs, extra_embeddings)
        embeddings = self._normalize(embeddings)
        return self._clustering_model.predict_proba(embeddings)

    def soft_predict(
        self,
        inputs: Iterable,
        extra_embeddings: Optional[Array] = None,
    ) -> Array:
        """
        For each input, predict which clusters the inputs are most likely to belong to.

        Parameters
        ----------
        inputs: Iterable
            An iterable of inputs.
        extra_embeddings: Optional[Array]
            An extra array of embeddings.

        Returns
        -------
        Array
            An array of bools determining whether an input is predicted to belong to a cluster or not.
        """
        probs = self.predict_proba(inputs=inputs, extra_embeddings=extra_embeddings)
        return probs > self._quantiles[None]

    def hard_predict(
        self,
        inputs: Iterable,
        extra_embeddings: Optional[Array] = None,
    ) -> Array:
        """
        For each input, predict the most likely cluster it belongs to.

        Parameters
        ----------
        inputs: Iterable
            An iterable of inputs.
        extra_embeddings: Optional[Array]
            An extra array of embeddings.

        Returns
        -------
        Array
            An array of bools determining whether an input is predicted to belong to a cluster or not.
            Exactly one True will be given for each input.
        """
        probs = self.predict_proba(inputs=inputs, extra_embeddings=extra_embeddings)

        bool_preds = np.zeros_like(probs, dtype=bool)
        bool_preds[np.arange(len(probs)), np.argmax(probs, axis=1)] = True
        return bool_preds

    @property
    def clustering_model(self):
        return self._clustering_model

    def _store_thresholds(self, probs, quantile_threshold: float) -> None:
        self._threshold = np.quantile(probs, quantile_threshold, axis=1)

    def _normalize(self, embeddings: Array) -> Array:
        if self._mean is None or self._std is None:
            raise ValueError("The `fit` method must be run first.")
        embeddings = np.copy(embeddings)
        embeddings -= self._mean
        embeddings /= self._std
        return embeddings

    def _get_concat_embeddings(
        self, inputs: Iterable, extra_embeddings: Optional[Array] = None
    ) -> Array:
        embeddings = self.embedding_manager.embed(inputs)
        if extra_embeddings is not None:
            if len(embeddings) != len(extra_embeddings):
                raise ValueError(
                    "`The total number of inputs must match the length of `extra_embeddings`."
                )
            embeddings = np.concatenate((embeddings, extra_embeddings), axis=1)
        return embeddings

    def _store_embeddings_stats(self, embeddings):
        self._mean = np.mean(embeddings, axis=0, keepdims=True)
        self._std = np.std(embeddings, axis=0, keepdims=True)

    def _fit_best_model(self, embeddings: Array, models: List) -> None:
        best_bic = np.inf
        for model in models:
            model = model.fit(embeddings)
            bic = model.bic(embeddings)
            if bic < best_bic:
                best_bic = bic
                best_model = model
        self._clustering_model = best_model
