import unittest

import jax.numpy as jnp
from flax.core import FrozenDict
from jax import random

from fortuna.data.loader import DataLoader
from fortuna.model.mlp import MLP
from fortuna.model.model_manager.regression import RegressionModelManager
from fortuna.output_calibrator.output_calib_manager.base import \
    OutputCalibManager
from fortuna.prob_model.joint.base import Joint
from fortuna.prob_model.likelihood.regression import RegressionLikelihood
from fortuna.prob_model.prior import IsotropicGaussianPrior
from fortuna.prob_output_layer.regression import RegressionProbOutputLayer
from tests.make_data import make_array_random_data


class TestJoints(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shape_inputs = (3,)
        self.n_inputs = 10
        self.output_dim = 2
        self.rng = random.PRNGKey(0)
        self.joint = Joint(
            prior=IsotropicGaussianPrior(),
            likelihood=RegressionLikelihood(
                model_manager=RegressionModelManager(
                    model=MLP(output_dim=self.output_dim),
                    likelihood_log_variance_model=MLP(output_dim=self.output_dim),
                ),
                prob_output_layer=RegressionProbOutputLayer(),
                output_calib_manager=OutputCalibManager(output_calibrator=None),
            ),
        )

        self.data_arr = DataLoader.from_array_data(
            make_array_random_data(
                n_data=self.n_inputs,
                shape_inputs=self.shape_inputs,
                output_dim=self.output_dim,
                output_type="continuous",
            )
        )

        self.params = FrozenDict(
            dict(
                model=self.joint.likelihood.model_manager.model.init(
                    self.rng, jnp.zeros((1,) + self.shape_inputs)
                ),
                lik_log_var=self.joint.likelihood.model_manager.likelihood_log_variance_model.init(
                    self.rng, jnp.zeros((1,) + self.shape_inputs)
                ),
            )
        )

    def test_lik_batched_log_prob(self):
        log_prob, aux = self.joint.log_prob(
            self.params, self.data_arr, return_aux=["outputs"]
        )
        assert jnp.array([log_prob]).shape == (1,)
        assert aux["outputs"].shape == (self.n_inputs, 2 * self.output_dim)

    def test_lik_log_prob(self):
        for batch in self.data_arr:
            log_prob, aux = self.joint.batched_log_prob(
                self.params, batch, n_data=batch[1].shape[0], return_aux=["outputs"]
            )
            assert jnp.array([log_prob]).shape == (1,)
            assert aux["outputs"].shape == (self.n_inputs, 2 * self.output_dim)
