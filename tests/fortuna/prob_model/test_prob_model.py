import unittest

from jax import random
import jax.numpy as jnp

from fortuna.data.loader import DataLoader
from fortuna.model.mlp import MLP
from fortuna.output_calibrator.classification import ClassificationTemperatureScaler
from fortuna.output_calibrator.regression import RegressionTemperatureScaler
from fortuna.prob_model.classification import ProbClassifier
from fortuna.prob_model.prior import IsotropicGaussianPrior
from fortuna.prob_model.regression import ProbRegressor
from tests.make_data import make_array_random_data


class TestProbModels(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_shape = (3, 4)
        self.output_dim = 2
        self.n_data = 100
        self.rng = random.PRNGKey(0)
        self.model = MLP(output_dim=self.output_dim)
        self.lik_log_var = MLP(output_dim=self.output_dim)
        self.reg_data = DataLoader.from_array_data(
            make_array_random_data(
                self.n_data, self.input_shape, self.output_dim, "continuous"
            )
        )
        self.class_data = DataLoader.from_array_data(
            make_array_random_data(
                self.n_data, self.input_shape, self.output_dim, "discrete"
            )
        )

    def test_prob_reg_init_params(self):
        prob_reg = ProbRegressor(
            model=self.model,
            likelihood_log_variance_model=self.lik_log_var,
            prior=IsotropicGaussianPrior(),
        )
        state = prob_reg.joint.init(self.input_shape)
        assert "model" in state.params
        assert "lik_log_var" in state.params
        assert "params" in state.params["model"]
        assert "params" in state.params["lik_log_var"]
        assert hasattr(state, "mutable")

    def test_prob_class_init_params(self):
        prob_class = ProbClassifier(model=self.model, prior=IsotropicGaussianPrior())
        state = prob_class.joint.init(self.input_shape)
        assert "model" in state.params
        assert "params" in state.params["model"]
        assert hasattr(state, "mutable")

    def test_temp_scaling_prob_reg_init_params(self):
        calib_prob_reg = ProbRegressor(
            model=self.model,
            likelihood_log_variance_model=self.lik_log_var,
            prior=IsotropicGaussianPrior(),
            output_calibrator=RegressionTemperatureScaler(),
        )
        state = calib_prob_reg.joint.init(self.input_shape)
        assert "model" in state.params
        assert "lik_log_var" in state.params
        assert "params" in state.params["model"]
        assert "params" in state.params["lik_log_var"]
        assert hasattr(state, "mutable")

    def test_temp_scaling_prob_class_init_params(self):
        calib_prob_class = ProbClassifier(
            model=self.model,
            prior=IsotropicGaussianPrior(),
            output_calibrator=ClassificationTemperatureScaler(),
        )
        state = calib_prob_class.joint.init(self.input_shape)
        assert "model" in state.params
        assert "params" in state.params["model"]
        assert hasattr(state, "mutable")

    def test_temp_scaling_prob_reg_init_calib_state(self):
        calib_prob_reg = ProbRegressor(
            model=self.model,
            likelihood_log_variance_model=self.model,
            prior=IsotropicGaussianPrior(),
            output_calibrator=RegressionTemperatureScaler(),
        )
        pms = calib_prob_reg.joint.init(self.input_shape)
        assert pms.calib_params["output_calibrator"]["params"]["log_temp"] == 0.0
        assert pms.calib_mutable["output_calibrator"] is None

    def test_temp_scaling_prob_class_init_calib_state(self):
        calib_prob_class = ProbClassifier(
            model=self.model,
            prior=IsotropicGaussianPrior(),
            output_calibrator=ClassificationTemperatureScaler(),
        )
        pms = calib_prob_class.joint.init(self.input_shape)
        assert pms.calib_params["output_calibrator"]["params"]["log_temp"] == 0.0
        assert pms.calib_mutable["output_calibrator"] is None

    def test_rng(self):
        prob_model = ProbClassifier(
            model=self.model,
            prior=IsotropicGaussianPrior(),
            output_calibrator=ClassificationTemperatureScaler(),
        )
        assert jnp.alltrue(prob_model.rng._rng == jnp.array([0.0, 0.0]))
        assert jnp.alltrue(prob_model.rng._rng == prob_model.posterior.rng._rng)
        assert jnp.alltrue(prob_model.rng._rng == prob_model.likelihood.rng._rng)
        assert jnp.alltrue(prob_model.rng._rng == prob_model.joint.rng._rng)
        assert jnp.alltrue(prob_model.rng._rng == prob_model.prior.rng._rng)
        assert jnp.alltrue(prob_model.rng._rng == prob_model.predictive.rng._rng)
        assert jnp.alltrue(
            prob_model.rng._rng == prob_model.output_calib_manager.rng._rng
        )
        prob_model.rng.get()
        assert jnp.alltrue(prob_model.rng._rng == prob_model.posterior.rng._rng)
        assert jnp.alltrue(prob_model.rng._rng == prob_model.likelihood.rng._rng)
        assert jnp.alltrue(prob_model.rng._rng == prob_model.joint.rng._rng)
        assert jnp.alltrue(prob_model.rng._rng == prob_model.prior.rng._rng)
        assert jnp.alltrue(prob_model.rng._rng == prob_model.predictive.rng._rng)
        assert jnp.alltrue(
            prob_model.rng._rng == prob_model.output_calib_manager.rng._rng
        )
        prob_model.posterior.rng.get()
        assert jnp.alltrue(prob_model.rng._rng == prob_model.posterior.rng._rng)
        assert jnp.alltrue(prob_model.rng._rng == prob_model.likelihood.rng._rng)
        assert jnp.alltrue(prob_model.rng._rng == prob_model.joint.rng._rng)
        assert jnp.alltrue(prob_model.rng._rng == prob_model.prior.rng._rng)
        assert jnp.alltrue(prob_model.rng._rng == prob_model.predictive.rng._rng)
        assert jnp.alltrue(
            prob_model.rng._rng == prob_model.output_calib_manager.rng._rng
        )
