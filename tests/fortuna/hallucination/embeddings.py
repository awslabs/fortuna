import unittest

from jax import random

from fortuna.data import InputsLoader
from fortuna.hallucination.embedding import EmbeddingManager


class TestEmbeddingsManager(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_inputs = 10
        self.n_features = 4
        self.n_reduced_features = 3
        self.inputs_loader = InputsLoader.from_array_inputs(
            random.normal(random.PRNGKey(0), shape=(self.n_inputs, self.n_features)),
            batch_size=2,
        )
        self.embedding_manager = EmbeddingManager(
            encoding_fn=lambda x: 1 - x,
            reduction_fn=lambda x: x[:, : self.n_reduced_features],
        )

    def test_get(self):
        embeddings = self.embedding_manager.get(self.inputs_loader)
        assert embeddings.shape == (self.n_inputs, self.n_reduced_features)
