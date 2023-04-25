import unittest

import jax.numpy as jnp
from flax.core import FrozenDict
from jax import random

from fortuna.data.loader import DataLoader, InputsLoader
from fortuna.model.mlp import MLP
from fortuna.model.model_manager.classification import \
    ClassificationModelManager
from fortuna.model.model_manager.regression import RegressionModelManager
from fortuna.output_calibrator.output_calib_manager.base import \
    OutputCalibManager
from fortuna.likelihood.classification import ClassificationLikelihood
from fortuna.likelihood.regression import RegressionLikelihood
from fortuna.prob_output_layer.classification import \
    ClassificationProbOutputLayer
from fortuna.prob_output_layer.regression import RegressionProbOutputLayer
from fortuna.utils.random import RandomNumberGenerator
from tests.make_data import (make_array_random_data,
                             make_generator_fun_random_data)


class TestLikelihoods(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shape_inputs = (3,)
        self.n_inputs = 10
        self.output_dim = 2
        self.n_batches = 2
        self.batch_size = 3
        self.rng = random.PRNGKey(0)
        rng = RandomNumberGenerator(seed=0)
        reg_prob_output_layer = RegressionProbOutputLayer()
        reg_prob_output_layer.rng = rng
        self.reg_lik = RegressionLikelihood(
            model_manager=RegressionModelManager(
                model=MLP(output_dim=self.output_dim),
                likelihood_log_variance_model=MLP(output_dim=self.output_dim),
            ),
            output_calib_manager=OutputCalibManager(output_calibrator=None),
            prob_output_layer=reg_prob_output_layer,
        )
        self.reg_lik.rng = rng
        class_prob_output_layer = ClassificationProbOutputLayer()
        class_prob_output_layer.rng = rng
        self.class_lik = ClassificationLikelihood(
            model_manager=ClassificationModelManager(
                model=MLP(output_dim=self.output_dim)
            ),
            output_calib_manager=OutputCalibManager(output_calibrator=None),
            prob_output_layer=class_prob_output_layer,
        )
        self.class_lik.rng = rng

        self.reg_data_arr = DataLoader.from_array_data(
            make_array_random_data(
                n_data=self.n_inputs,
                shape_inputs=self.shape_inputs,
                output_dim=self.output_dim,
                output_type="continuous",
            )
        )
        self.reg_inputs_arr = InputsLoader.from_data_loader(self.reg_data_arr)

        self.reg_data_gen_fun = DataLoader.from_callable_iterable(
            make_generator_fun_random_data(
                n_batches=self.n_batches,
                batch_size=self.batch_size,
                shape_inputs=self.shape_inputs,
                output_dim=self.output_dim,
                output_type="continuous",
            )
        )
        self.reg_inputs_gen_fun = InputsLoader.from_data_loader(self.reg_data_gen_fun)

        self.class_data_arr = DataLoader.from_array_data(
            make_array_random_data(
                n_data=self.n_inputs,
                shape_inputs=self.shape_inputs,
                output_dim=self.output_dim,
                output_type="discrete",
            )
        )
        self.class_inputs_arr = InputsLoader.from_data_loader(self.class_data_arr)

        self.class_data_gen_fun = DataLoader.from_callable_iterable(
            make_generator_fun_random_data(
                n_batches=self.n_batches,
                batch_size=self.batch_size,
                shape_inputs=self.shape_inputs,
                output_dim=self.output_dim,
                output_type="discrete",
            )
        )
        self.class_inputs_gen_fun = InputsLoader.from_data_loader(
            self.class_data_gen_fun
        )

    def test_lik_batched_log_joint_prob(self):
        params = FrozenDict(
            dict(
                model=self.reg_lik.model_manager.model.init(
                    self.rng, jnp.zeros((1,) + self.shape_inputs)
                ),
                lik_log_var=self.reg_lik.model_manager.likelihood_log_variance_model.init(
                    self.rng, jnp.zeros((1,) + self.shape_inputs)
                ),
            )
        )

        for batch_data in self.reg_data_arr:
            batched_log_joint_prob1 = self.reg_lik._batched_log_joint_prob(
                params, batch_data, n_data=batch_data[1].shape[0]
            )
            batched_log_joint_prob2 = self.reg_lik._batched_log_joint_prob(
                params, batch_data, n_data=2 * batch_data[1].shape[0]
            )
            assert jnp.allclose(batched_log_joint_prob2, 2 * batched_log_joint_prob1)
            assert jnp.array([batched_log_joint_prob1]).shape == (1,)

            _, aux = self.reg_lik._batched_log_joint_prob(
                params,
                batch_data,
                n_data=batch_data[1].shape[0],
                return_aux=["outputs"],
            )
            assert aux["outputs"].shape == (self.n_inputs, 2 * self.output_dim)

    def test_lik_log_joint_prob(self):
        params = FrozenDict(
            dict(
                model=self.reg_lik.model_manager.model.init(
                    self.rng, jnp.zeros((1,) + self.shape_inputs)
                ),
                lik_log_var=self.reg_lik.model_manager.likelihood_log_variance_model.init(
                    self.rng, jnp.zeros((1,) + self.shape_inputs)
                ),
            )
        )

        log_probs = self.reg_lik.log_prob(params, self.reg_data_arr)
        assert log_probs.shape == (self.n_inputs,)

        log_probs = self.reg_lik.log_prob(params, self.reg_data_gen_fun)
        assert log_probs.shape == (self.n_batches * self.batch_size,)

    def test_sample(self):
        params = FrozenDict(
            dict(
                model=self.reg_lik.model_manager.model.init(
                    self.rng, jnp.zeros((1,) + self.shape_inputs)
                ),
                lik_log_var=self.reg_lik.model_manager.likelihood_log_variance_model.init(
                    self.rng, jnp.zeros((1,) + self.shape_inputs)
                ),
            )
        )

        samples = self.reg_lik.sample(10, params, self.reg_inputs_arr)
        assert samples.shape == (10, self.n_inputs, self.output_dim)

        params = FrozenDict(
            dict(
                model=self.class_lik.model_manager.model.init(
                    self.rng, jnp.zeros((1,) + self.shape_inputs)
                ),
            )
        )

        samples = self.class_lik.sample(10, params, self.class_inputs_arr)
        assert samples.shape == (10, self.n_inputs)

    def test_reg_stats(self):
        params = FrozenDict(
            dict(
                model=self.reg_lik.model_manager.model.init(
                    self.rng, jnp.zeros((1,) + self.shape_inputs)
                ),
                lik_log_var=self.reg_lik.model_manager.likelihood_log_variance_model.init(
                    self.rng, jnp.zeros((1,) + self.shape_inputs)
                ),
            )
        )

        assert self.reg_lik.mean(params, self.reg_inputs_arr).shape == (
            self.n_inputs,
            self.output_dim,
        )
        assert self.reg_lik.mean(params, self.reg_inputs_gen_fun).shape == (
            self.batch_size * self.n_batches,
            self.output_dim,
        )

        assert self.reg_lik.mode(params, self.reg_inputs_arr).shape == (
            self.n_inputs,
            self.output_dim,
        )
        assert self.reg_lik.mode(params, self.reg_inputs_gen_fun).shape == (
            self.batch_size * self.n_batches,
            self.output_dim,
        )
        assert jnp.allclose(
            self.reg_lik.mode(params, self.reg_inputs_arr),
            self.reg_lik.mean(params, self.reg_inputs_arr),
        )

        variance = self.reg_lik.variance(params, self.reg_inputs_arr)
        assert variance.shape == (self.n_inputs, self.output_dim)
        assert self.reg_lik.variance(params, self.reg_inputs_gen_fun).shape == (
            self.batch_size * self.n_batches,
            self.output_dim,
        )
        assert (variance >= 0).all()

        assert self.reg_lik.entropy(params, self.reg_inputs_arr).shape == (
            self.n_inputs,
        )
        assert self.reg_lik.entropy(params, self.reg_inputs_gen_fun).shape == (
            self.batch_size * self.n_batches,
        )

        assert self.reg_lik.quantile(
            jnp.array([0.3, 0.6]),
            params=params,
            inputs_loader=self.reg_inputs_arr,
        ).shape == (2, self.n_inputs, self.output_dim)

    def test_class_stats(self):
        params = FrozenDict(
            dict(
                model=self.class_lik.model_manager.model.init(
                    self.rng, jnp.zeros((1,) + self.shape_inputs)
                )
            )
        )

        mean = self.class_lik.mean(params, self.class_inputs_arr)
        assert mean.shape == (self.n_inputs, self.output_dim)
        assert self.class_lik.mean(params, self.class_inputs_gen_fun).shape == (
            self.batch_size * self.n_batches,
            self.output_dim,
        )
        assert (mean >= 0).all() and (mean <= 1).all()

        mode = self.class_lik.mode(params, self.class_inputs_arr)
        assert mode.shape == (self.n_inputs,)
        assert self.class_lik.mode(params, self.class_inputs_gen_fun).shape == (
            self.batch_size * self.n_batches,
        )
        assert all(mode) in jnp.arange(self.output_dim)

        variance = self.class_lik.variance(params, self.class_inputs_arr)
        assert variance.shape == (self.n_inputs, self.output_dim)
        assert self.class_lik.variance(params, self.class_inputs_gen_fun).shape == (
            self.batch_size * self.n_batches,
            self.output_dim,
        )
        assert (variance >= 0).all()

        assert self.class_lik.entropy(params, self.class_inputs_arr).shape == (
            self.n_inputs,
        )
        assert self.class_lik.entropy(params, self.class_inputs_gen_fun).shape == (
            self.batch_size * self.n_batches,
        )
