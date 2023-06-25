import tempfile
import unittest

from jax import random
import jax.numpy as jnp
import optax

from fortuna.data.loader import (
    DataLoader,
    InputsLoader,
)
from fortuna.prob_model.classification import ProbClassifier
from fortuna.prob_model.posterior.map.map_approximator import MAPPosteriorApproximator
from fortuna.prob_model.regression import ProbRegressor
from tests.make_data import make_array_random_data
from tests.make_model import MyModel


class TestPredictives(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shape_inputs = (3,)
        self.n_inputs = 10
        self.output_dim = 2
        self.n_batches = 2
        self.batch_size = 3
        self.n_post_samples = 4

        self.reg_data_loader = DataLoader.from_array_data(
            make_array_random_data(
                n_data=self.n_inputs,
                shape_inputs=self.shape_inputs,
                output_dim=self.output_dim,
                output_type="continuous",
            )
        )
        self.reg_inputs_loader = InputsLoader.from_data_loader(self.reg_data_loader)

        self.class_data_loader = DataLoader.from_array_data(
            make_array_random_data(
                n_data=self.n_inputs,
                shape_inputs=self.shape_inputs,
                output_dim=self.output_dim,
                output_type="discrete",
            )
        )
        self.class_inputs_loader = InputsLoader.from_data_loader(self.class_data_loader)

        self.rng = random.PRNGKey(0)

        self.prob_class = ProbClassifier(
            model=MyModel(self.output_dim),
            posterior_approximator=MAPPosteriorApproximator(),
        )
        self.prob_reg = ProbRegressor(
            model=MyModel(self.output_dim),
            likelihood_log_variance_model=MyModel(self.output_dim),
            posterior_approximator=MAPPosteriorApproximator(),
        )

    def test_pred_stats(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            status = self.prob_class.posterior.fit(
                train_data_loader=self.class_data_loader,
                save_checkpoint_dir=tmp_dir,
                optimizer=optax.adam(1e-2),
                n_epochs=2,
            )
            status = self.prob_reg.posterior.fit(
                train_data_loader=self.reg_data_loader,
                save_checkpoint_dir=tmp_dir,
                optimizer=optax.adam(1e-2),
                n_epochs=2,
            )
            log_probs = self.prob_class.predictive.log_prob(
                self.class_data_loader,
                n_posterior_samples=self.n_post_samples,
            )
            assert log_probs.shape == (self.n_inputs,)

            log_probs = self.prob_reg.predictive.log_prob(
                self.reg_data_loader,
                n_posterior_samples=self.n_post_samples,
            )
            assert log_probs.shape == (self.n_inputs,)

            sample = self.prob_class.predictive.sample(
                self.class_inputs_loader, n_target_samples=self.n_post_samples
            )
            assert sample.shape == (
                self.n_post_samples,
                self.n_inputs,
            )

            sample = self.prob_reg.predictive.sample(
                self.reg_inputs_loader, n_target_samples=self.n_post_samples
            )
            assert sample.shape == (self.n_post_samples, self.n_inputs, self.output_dim)

            assert self.prob_reg.predictive.mean(
                self.reg_inputs_loader,
                n_posterior_samples=self.n_post_samples,
            ).shape == (self.n_inputs, self.output_dim)

            assert self.prob_class.predictive.mean(
                self.class_inputs_loader,
                n_posterior_samples=self.n_post_samples,
            ).shape == (self.n_inputs, self.output_dim)

            assert self.prob_reg.predictive.mode(
                self.reg_inputs_loader,
                n_posterior_samples=self.n_post_samples,
            ).shape == (self.n_inputs, self.output_dim)

            assert self.prob_class.predictive.mode(
                self.class_inputs_loader,
                n_posterior_samples=self.n_post_samples,
            ).shape == (self.n_inputs,)

            variance = self.prob_reg.predictive.variance(
                self.reg_inputs_loader,
                n_posterior_samples=self.n_post_samples,
            )
            assert variance.shape == (self.n_inputs, self.output_dim)
            assert (variance >= 0).all()

            variance = self.prob_class.predictive.variance(
                self.class_inputs_loader,
                n_posterior_samples=self.n_post_samples,
            )
            assert variance.shape == (self.n_inputs, self.output_dim)
            assert (variance >= 0).all()

            aleatoric_variance = self.prob_reg.predictive.aleatoric_variance(
                self.reg_inputs_loader,
                n_posterior_samples=self.n_post_samples,
            )
            assert aleatoric_variance.shape == (self.n_inputs, self.output_dim)
            assert (aleatoric_variance >= 0).all()

            aleatoric_variance = self.prob_class.predictive.aleatoric_variance(
                self.class_inputs_loader,
                n_posterior_samples=self.n_post_samples,
            )
            assert aleatoric_variance.shape == (self.n_inputs, self.output_dim)
            assert (aleatoric_variance >= 0).all()

            epistemic_variance = self.prob_reg.predictive.epistemic_variance(
                self.reg_inputs_loader,
                n_posterior_samples=self.n_post_samples,
            )
            assert epistemic_variance.shape == (self.n_inputs, self.output_dim)
            assert (epistemic_variance >= 0).all()

            epistemic_variance = self.prob_class.predictive.epistemic_variance(
                self.class_inputs_loader,
                n_posterior_samples=self.n_post_samples,
            )
            assert epistemic_variance.shape == (self.n_inputs, self.output_dim)
            assert (epistemic_variance >= 0).all()

            assert self.prob_reg.predictive.entropy(
                self.reg_inputs_loader,
                n_posterior_samples=self.n_post_samples,
            ).shape == (self.n_inputs,)

            assert self.prob_class.predictive.entropy(
                self.class_inputs_loader,
                n_posterior_samples=self.n_post_samples,
            ).shape == (self.n_inputs,)

            assert self.prob_reg.predictive.aleatoric_entropy(
                self.reg_inputs_loader,
                n_posterior_samples=self.n_post_samples,
            ).shape == (self.n_inputs,)

            assert self.prob_class.predictive.aleatoric_entropy(
                self.class_inputs_loader,
                n_posterior_samples=self.n_post_samples,
            ).shape == (self.n_inputs,)

            assert self.prob_reg.predictive.epistemic_entropy(
                self.reg_inputs_loader,
                n_posterior_samples=self.n_post_samples,
            ).shape == (self.n_inputs,)

            assert self.prob_class.predictive.epistemic_entropy(
                self.class_inputs_loader,
                n_posterior_samples=self.n_post_samples,
            ).shape == (self.n_inputs,)

            assert self.prob_reg.predictive.quantile(
                jnp.array([0.3, 0.6]),
                inputs_loader=self.reg_inputs_loader,
            ).shape == (2, self.n_inputs, self.output_dim)
