import unittest

from jax import random
import jax.numpy as jnp
import optax

from fortuna.data.loader import DataLoader
from fortuna.model.mlp import MLP
from fortuna.output_calibrator.classification import ClassificationTemperatureScaler
from fortuna.prob_model.classification import ProbClassifier
from fortuna.prob_model.posterior.map.map_approximator import MAPPosteriorApproximator
from tests.make_data import make_array_random_data


class TestCalibrators(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rng = random.PRNGKey(0)
        self.n_inputs = 6
        self.shape_inputs = (4,)
        self.output_dim = 2
        self.checkpoint_dir = "logs"

        self.class_data_loader = DataLoader.from_array_data(
            make_array_random_data(
                n_data=self.n_inputs,
                shape_inputs=self.shape_inputs,
                output_dim=self.output_dim,
                output_type="discrete",
            )
        )

    def test_calibrate_prob_model(self):
        prob_model = ProbClassifier(
            model=MLP(self.output_dim),
            output_calibrator=ClassificationTemperatureScaler(),
            posterior_approximator=MAPPosteriorApproximator(),
        )
        status = prob_model.posterior.fit(
            train_data_loader=self.class_data_loader,
            optimizer=optax.adam(1e-2),
            n_epochs=2,
        )
        status = prob_model.calibrate(self.class_data_loader)
        s = prob_model.posterior.state.get()
        assert s.calib_params["output_calibrator"]["params"]["log_temp"].shape == (1,)
        assert s.calib_params["output_calibrator"]["params"]["log_temp"] != jnp.array(
            [0.0]
        )
