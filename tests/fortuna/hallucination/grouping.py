import unittest

from jax import random
import numpy as np
from sklearn.mixture import GaussianMixture

from fortuna.data import InputsLoader
from fortuna.hallucination.embedding import EmbeddingManager
from fortuna.hallucination.grouping.clustering.base import GroupingModel


class GroupingModelTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_inputs = 10
        self.n_features = 4
        self.n_reduced_features = 3
        self.n_extra_features = 5
        self.inputs_loader = InputsLoader.from_array_inputs(
            random.normal(random.PRNGKey(0), shape=(self.n_inputs, self.n_features)),
            batch_size=2,
        )
        self.grouping_model = GroupingModel(
            embedding_manager=EmbeddingManager(
                encoding_fn=lambda x: 1 - x,
                reduction_fn=lambda x: x[:, : self.n_reduced_features],
            )
        )
        self.extra_embeddings = random.normal(
            random.PRNGKey(0), shape=(self.n_inputs, self.n_extra_features)
        )
        self.clustering_models = [GaussianMixture(n_components=i) for i in range(2, 4)]

    def test_all(self):
        self.grouping_model.fit(
            inputs_loader=self.inputs_loader,
            extra_embeddings=None,
            clustering_models=self.clustering_models,
        )
        self._check_shape_types(extra_embeddings=None)

        self.grouping_model.fit(
            inputs_loader=self.inputs_loader,
            extra_embeddings=self.extra_embeddings,
            clustering_models=self.clustering_models,
        )
        self._check_shape_types(extra_embeddings=self.extra_embeddings)

        with self.assertRaises(ValueError):
            self.grouping_model.fit(
                inputs_loader=self.inputs_loader,
                extra_embeddings=None,
                clustering_models=[],
            )

        with self.assertRaises(ValueError):
            self.grouping_model.fit(
                inputs_loader=self.inputs_loader,
                extra_embeddings=np.zeros((self.n_inputs + 1, 2)),
                clustering_models=[],
            )

    def _check_shape_types(self, extra_embeddings):
        probs = self.grouping_model.predict_proba(
            inputs_loader=self.inputs_loader, extra_embeddings=extra_embeddings
        )
        hard_preds = self.grouping_model.hard_predict(
            inputs_loader=self.inputs_loader, extra_embeddings=extra_embeddings
        )
        soft_preds = self.grouping_model.hard_predict(
            inputs_loader=self.inputs_loader, extra_embeddings=extra_embeddings
        )
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
