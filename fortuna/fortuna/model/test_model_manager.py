import unittest

import jax.numpy as jnp
from flax.core import FrozenDict
from fortuna.model.mlp import MLP
from fortuna.model.model_manager.classification import \
    ClassificationModelManager
from fortuna.model.model_manager.regression import RegressionModelManager
from jax import random
from tests.make_data import make_array_random_inputs


class TestModelManagers(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shape_inputs = (4,)
        self.output_dim = 2
        self.n_inputs = 10
        self.rng = random.PRNGKey(0)
        self.model = MLP(output_dim=self.output_dim)
        self.lik_log_var = MLP(output_dim=self.output_dim)

    def test_classifier_model_manager_apply(self):
        classifier_model_manager = ClassificationModelManager(self.model)
        params = FrozenDict(
            dict(model=self.model.init(self.rng, jnp.zeros((2,) + self.shape_inputs)))
        )

        inputs = make_array_random_inputs(
            n_inputs=self.n_inputs, shape_inputs=self.shape_inputs
        )
        assert classifier_model_manager.apply(params, inputs).shape == (
            self.n_inputs,
            self.output_dim,
        )

    def test_regressor_model_manager_apply(self):
        regressor_model_manager = RegressionModelManager(self.model, self.lik_log_var)
        params = FrozenDict(
            dict(
                model=self.model.init(self.rng, jnp.zeros((2,) + self.shape_inputs)),
                lik_log_var=self.model.init(
                    self.rng, jnp.zeros((2,) + self.shape_inputs)
                ),
            )
        )

        inputs = make_array_random_inputs(
            n_inputs=self.n_inputs, shape_inputs=self.shape_inputs
        )
        assert regressor_model_manager.apply(params, inputs).shape == (
            self.n_inputs,
            2 * self.output_dim,
        )
