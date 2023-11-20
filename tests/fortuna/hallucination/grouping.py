import unittest

from jax import random
import numpy as np
from sklearn.mixture import GaussianMixture

from fortuna.hallucination.grouping.clustering.base import GroupingModel


class GroupingModelTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_inputs = 10
        self.n_features = 4
        self.n_reduced_features = 3
        self.embeddings = random.normal(
            random.PRNGKey(0), shape=(self.n_inputs, self.n_features)
        )
        self.grouping_model = GroupingModel()
        self.extra_embeddings = random.normal(
            random.PRNGKey(0), shape=(self.n_inputs, self.n_extra_features)
        )
        self.clustering_models = [GaussianMixture(n_components=i) for i in range(2, 4)]

    def test_all(self):
        self.grouping_model.fit(
            embeddings=self.embeddings,
            clustering_models=self.clustering_models,
        )
        self._check_shape_types()

        self.grouping_model.fit(
            embeddings=self.embeddings,
            clustering_models=self.clustering_models,
        )
        self._check_shape_types()

        with self.assertRaises(ValueError):
            self.grouping_model.fit(
                embeddings=self.embeddings,
                clustering_models=[],
            )

        with self.assertRaises(ValueError):
            self.grouping_model.fit(
                embeddings=self.embeddings,
                clustering_models=[],
            )

    def _check_shape_types(self):
        probs = self.grouping_model.predict_proba(embeddings=self.embeddings)
        hard_preds = self.grouping_model.hard_predict(embeddings=self.embeddings)
        soft_preds = self.grouping_model.hard_predict(embeddings=self.embeddings)
        assert probs.shape == (
            self.n_inputs,
            self.grouping_model._clustering_model.n_components,
        )
        assert soft_preds.shape == (
            self.n_inputs,
            self.grouping_model._clustering_model.n_components,
        )
        assert hard_preds.shape == (
            self.n_inputs,
            self.grouping_model._clustering_model.n_components,
        )
        assert soft_preds.dtype == bool
        assert hard_preds.dtype == bool
        assert np.allclose(hard_preds.sum(1), 1)
