import unittest

import jax.numpy as jnp
import jax.scipy as jsp
from fortuna.prob_output_layer.classification import \
    ClassificationProbOutputLayer
from fortuna.prob_output_layer.regression import RegressionProbOutputLayer
from fortuna.utils.random import RandomNumberGenerator
from jax import random


class TestProbOutputLayers(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim_outputs = 2
        self.n_inputs = 10
        self.n_samples = 11
        rng = RandomNumberGenerator(seed=0)
        self.rng_outputs = random.PRNGKey(0)
        self.rng_targets = random.PRNGKey(1)
        self.rng_samples = random.PRNGKey(2)
        self.reg_prob_output_layer = RegressionProbOutputLayer()
        self.reg_prob_output_layer.rng = rng
        self.class_prob_output_layer = ClassificationProbOutputLayer()
        self.class_prob_output_layer.rng = rng

    def test_reg_prob_output_layer_logprob(self):
        outputs = random.normal(
            self.rng_outputs, shape=(self.n_inputs, 2 * self.dim_outputs)
        )
        targets = random.normal(
            self.rng_targets, shape=(self.n_inputs, self.dim_outputs)
        )
        assert self.reg_prob_output_layer.log_prob(outputs, targets).shape == (
            self.n_inputs,
        )

        outputs = jnp.ones((1, 2))
        targets = jnp.zeros((1, 1))
        assert jnp.allclose(
            self.reg_prob_output_layer.log_prob(outputs, targets),
            -0.5 * (jnp.log(2 * jnp.pi) + 1 + jnp.exp(-1)),
        )

    def test_reg_prob_output_layer_predict(self):
        outputs = random.normal(
            self.rng_outputs, shape=(self.n_inputs, 2 * self.dim_outputs)
        )

        assert self.reg_prob_output_layer.predict(outputs).shape == (
            self.n_inputs,
            self.dim_outputs,
        )

    def test_reg_prob_output_layer_sample(self):
        outputs = random.normal(
            self.rng_outputs, shape=(self.n_inputs, 2 * self.dim_outputs)
        )
        assert self.reg_prob_output_layer.sample(self.n_samples, outputs).shape == (
            self.n_samples,
            self.n_inputs,
            self.dim_outputs,
        )

    def test_class_prob_output_layer_logprob(self):
        outputs = random.normal(
            self.rng_outputs, shape=(self.n_inputs, self.dim_outputs)
        )
        targets = random.choice(
            self.rng_targets, self.dim_outputs, shape=(self.n_inputs,)
        )
        assert self.class_prob_output_layer.log_prob(outputs, targets).shape == (
            self.n_inputs,
        )

        outputs = jnp.ones((1, 2))
        targets = jnp.zeros(1)
        assert jnp.allclose(
            self.class_prob_output_layer.log_prob(outputs, targets),
            outputs[0] - jsp.special.logsumexp(outputs, -1),
        )

    def test_class_prob_output_layer_predict(self):
        outputs = random.normal(
            self.rng_outputs, shape=(self.n_inputs, 2 * self.dim_outputs)
        )

        assert self.class_prob_output_layer.predict(outputs).shape == (self.n_inputs,)

    def test_class_prob_output_layer_sample(self):
        outputs = random.normal(
            self.rng_outputs, shape=(self.n_inputs, 2 * self.dim_outputs)
        )
        assert self.class_prob_output_layer.sample(self.n_samples, outputs).shape == (
            self.n_samples,
            self.n_inputs,
        )
