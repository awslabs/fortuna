import logging
import tempfile
import unittest

import jax.numpy as jnp

from fortuna.calib_model.calib_config.base import CalibConfig
from fortuna.calib_model.calib_config.checkpointer import CalibCheckpointer
from fortuna.calib_model.calib_config.monitor import CalibMonitor
from fortuna.calib_model.calib_config.optimizer import CalibOptimizer
from fortuna.calib_model.classification import CalibClassifier
from fortuna.calib_model.regression import CalibRegressor
from fortuna.data.loader import DataLoader
from fortuna.metric.classification import accuracy, brier_score
from fortuna.metric.regression import rmse
from fortuna.model.mlp import MLP
from fortuna.output_calibrator.regression import RegressionTemperatureScaler
from fortuna.prob_model.classification import ProbClassifier
from fortuna.prob_model.fit_config import FitConfig, FitMonitor
from fortuna.prob_model.fit_config.optimizer import FitOptimizer
from fortuna.prob_model.posterior.map.map_approximator import \
    MAPPosteriorApproximator
from fortuna.prob_model.prior import IsotropicGaussianPrior
from fortuna.prob_model.regression import ProbRegressor
from tests.make_data import make_array_random_data
from tests.make_model import MyModel

logging.basicConfig(level=logging.INFO)


def brier(dummy, p, y):
    return brier_score(p, y)


def standard_error(m, v, y):
    return jnp.sum((y - m) ** 2 / v) / m.shape[0]


class TestApproximations(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prob_class = ProbClassifier(
            model=MyModel(2), prior=IsotropicGaussianPrior()
        )

        self.reg_input_shape = (3,)
        self.reg_output_dim = 2
        self.class_input_shape = (2,)
        self.class_output_dim = 2
        bs = 32
        x, y = make_array_random_data(
            n_data=100,
            shape_inputs=self.reg_input_shape,
            output_dim=self.reg_output_dim,
            output_type="continuous",
        )
        x /= x.max(0)
        y /= y.max(0)
        reg_train_data = x, y
        reg_val_data = make_array_random_data(
            n_data=100,
            shape_inputs=self.reg_input_shape,
            output_dim=self.reg_output_dim,
            output_type="continuous",
        )
        reg_train_data = [
            (reg_train_data[0][i : i + bs], reg_train_data[1][i : i + bs])
            for i in range(0, len(reg_train_data[0]), bs)
        ]
        reg_val_data = [
            (reg_val_data[0][i : i + bs], reg_val_data[1][i : i + bs])
            for i in range(0, len(reg_val_data[0]), bs)
        ]
        self.reg_train_data_loader = DataLoader.from_iterable(reg_train_data)
        self.reg_val_data_loader = DataLoader.from_iterable(reg_val_data)

        class_train_data = make_array_random_data(
            n_data=100,
            shape_inputs=self.class_input_shape,
            output_dim=self.class_output_dim,
            output_type="discrete",
        )
        class_val_data = make_array_random_data(
            n_data=100,
            shape_inputs=self.class_input_shape,
            output_dim=self.class_output_dim,
            output_type="discrete",
        )
        class_train_data = [
            (class_train_data[0][i : i + bs], class_train_data[1][i : i + bs])
            for i in range(0, len(class_train_data[0]), bs)
        ]
        class_val_data = [
            (class_val_data[0][i : i + bs], class_val_data[1][i : i + bs])
            for i in range(0, len(class_val_data[0]), bs)
        ]
        self.class_train_data_loader = DataLoader.from_iterable(class_train_data)
        self.class_val_data_loader = DataLoader.from_iterable(class_val_data)

        self.class_fit_config_nodir_nodump = FitConfig(
            optimizer=FitOptimizer(n_epochs=3), monitor=FitMonitor(metrics=(accuracy,))
        )
        self.reg_fit_config_nodir_nodump = FitConfig(
            optimizer=FitOptimizer(n_epochs=3), monitor=FitMonitor(metrics=(rmse,))
        )
        self.calib_config_dir_nodump = lambda directory, metric: CalibConfig(
            optimizer=CalibOptimizer(n_epochs=3),
            monitor=CalibMonitor(metrics=(metric,)),
            checkpointer=CalibCheckpointer(save_checkpoint_dir=directory),
        )
        self.calib_config_dir_dump = lambda directory, metric: CalibConfig(
            optimizer=CalibOptimizer(n_epochs=3),
            monitor=CalibMonitor(metrics=(metric,)),
            checkpointer=CalibCheckpointer(
                save_checkpoint_dir=directory, dump_state=True
            ),
        )
        self.calib_config_restore = lambda directory, metric: CalibConfig(
            optimizer=CalibOptimizer(n_epochs=3),
            monitor=CalibMonitor(metrics=(metric,)),
            checkpointer=CalibCheckpointer(restore_checkpoint_path=directory),
        )

    def test_dryrun_reg_map(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            prob_reg = ProbRegressor(
                model=MLP(self.reg_output_dim),
                likelihood_log_variance_model=MyModel(self.reg_output_dim),
                posterior_approximator=MAPPosteriorApproximator(),
                output_calibrator=RegressionTemperatureScaler(),
            )
            # no save dir, no dump
            train_status = prob_reg.train(
                train_data_loader=self.reg_train_data_loader,
                calib_data_loader=self.reg_val_data_loader,
                fit_config=self.reg_fit_config_nodir_nodump,
            )

            state = prob_reg.posterior.state.get()
            outputs = prob_reg.model_manager.apply(
                params=state.params,
                inputs=self.reg_val_data_loader.to_array_inputs(),
                mutable=state.mutable,
            )
            targets = self.reg_val_data_loader.to_array_targets()

            # calibrate from initialized state, save checkpoint
            calib_model = CalibRegressor()
            calib_status = calib_model.calibrate(
                calib_outputs=outputs,
                calib_targets=targets,
                val_outputs=outputs,
                val_targets=targets,
                calib_config=self.calib_config_dir_nodump(tmp_dir, standard_error),
            )

            # calibrate from restored checkpoint
            calib_status = calib_model.calibrate(
                calib_outputs=outputs,
                calib_targets=targets,
                val_outputs=outputs,
                val_targets=targets,
                calib_config=self.calib_config_restore(tmp_dir, standard_error),
            )

            # calibrate from restored checkpoint, save checkpoint and dump
            calib_status = calib_model.calibrate(
                calib_outputs=outputs,
                calib_targets=targets,
                val_outputs=outputs,
                val_targets=targets,
                calib_config=self.calib_config_dir_dump(tmp_dir, standard_error),
            )

            # load state
            calib_model.load_state(checkpoint_path=tmp_dir)

            # save state
            calib_model.save_state(checkpoint_path=tmp_dir)

    def test_dryrun_class_map(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            prob_class = ProbClassifier(
                model=MLP(self.class_output_dim),
                posterior_approximator=MAPPosteriorApproximator(),
                output_calibrator=RegressionTemperatureScaler(),
            )
            # no save dir, no dump
            train_status = prob_class.train(
                train_data_loader=self.class_train_data_loader,
                calib_data_loader=self.class_val_data_loader,
                fit_config=self.class_fit_config_nodir_nodump,
            )

            state = prob_class.posterior.state.get()
            outputs = prob_class.model_manager.apply(
                params=state.params,
                inputs=self.class_val_data_loader.to_array_inputs(),
                mutable=state.mutable,
            )
            targets = self.class_val_data_loader.to_array_targets()

            # calibrate from initialized state, save checkpoint
            calib_model = CalibClassifier()
            calib_status = calib_model.calibrate(
                calib_outputs=outputs,
                calib_targets=targets,
                val_outputs=outputs,
                val_targets=targets,
                calib_config=self.calib_config_dir_nodump(tmp_dir, brier),
            )

            # calibrate from restored checkpoint
            calib_status = calib_model.calibrate(
                calib_outputs=outputs,
                calib_targets=targets,
                val_outputs=outputs,
                val_targets=targets,
                calib_config=self.calib_config_restore(tmp_dir, brier),
            )

            # calibrate from restored checkpoint, save checkpoint and dump
            calib_status = calib_model.calibrate(
                calib_outputs=outputs,
                calib_targets=targets,
                val_outputs=outputs,
                val_targets=targets,
                calib_config=self.calib_config_dir_dump(tmp_dir, brier),
            )

            # load state
            calib_model.load_state(checkpoint_path=tmp_dir)

            # save state
            calib_model.save_state(checkpoint_path=tmp_dir)
