import logging
import tempfile
import unittest

from fortuna.calib_config.base import CalibConfig
from fortuna.calib_config.checkpointer import CalibCheckpointer
from fortuna.calib_model.classification import CalibClassifier
from fortuna.calib_model.regression import CalibRegressor
from fortuna.data.loader import DataLoader
from fortuna.metric.classification import accuracy
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
from sklearn.datasets import make_moons, make_regression
from tests.make_model import MyModel

logging.basicConfig(level=logging.INFO)


class TestApproximations(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prob_class = ProbClassifier(
            model=MyModel(2), prior=IsotropicGaussianPrior()
        )

        self.reg_input_shape = (3,)
        self.reg_output_dim = 2
        bs = 32
        x, y = make_regression(
            n_samples=100,
            n_features=self.reg_input_shape[0],
            n_targets=self.reg_output_dim,
            random_state=0,
        )
        x /= x.max(0)
        y /= y.max(0)
        reg_train_data = x, y
        reg_val_data = make_regression(
            n_samples=10,
            n_features=self.reg_input_shape[0],
            n_targets=self.reg_output_dim,
            random_state=1,
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

        self.class_input_shape = (2,)
        self.class_output_dim = 2
        class_train_data = make_moons(n_samples=100, noise=0.07, random_state=0)
        class_val_data = make_moons(n_samples=10, noise=0.07, random_state=1)
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

            # calibrate from initialized state
            calib_model = CalibRegressor()
            calib_status = calib_model.calibrate(
                outputs,
                self.reg_val_data_loader.to_array_targets(),
                calib_config=CalibConfig(
                    checkpointer=CalibCheckpointer(save_state_path=tmp_dir)
                ),
            )

            # calibrate from current state
            calib_status = calib_model.calibrate(
                outputs,
                self.reg_val_data_loader.to_array_targets(),
                calib_config=CalibConfig(
                    checkpointer=CalibCheckpointer(start_from_current_state=True)
                ),
            )

            calib_status = calib_model.calibrate(
                outputs,
                self.reg_val_data_loader.to_array_targets(),
                calib_config=CalibConfig(
                    checkpointer=CalibCheckpointer(start_from_current_state=True)
                ),
            )

            calib_status = calib_model.calibrate(
                outputs,
                self.reg_val_data_loader.to_array_targets(),
                calib_config=CalibConfig(
                    checkpointer=CalibCheckpointer(restore_checkpoint_path=tmp_dir)
                ),
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

            # calibrate from initialized state
            calib_model = CalibClassifier()
            calib_status = calib_model.calibrate(
                outputs,
                self.class_val_data_loader.to_array_targets(),
                calib_config=CalibConfig(
                    checkpointer=CalibCheckpointer(save_state_path=tmp_dir)
                ),
            )

            # calibrate from current state
            calib_status = calib_model.calibrate(
                outputs,
                self.class_val_data_loader.to_array_targets(),
                calib_config=CalibConfig(
                    checkpointer=CalibCheckpointer(start_from_current_state=True)
                ),
            )

            calib_status = calib_model.calibrate(
                outputs,
                self.class_val_data_loader.to_array_targets(),
                calib_config=CalibConfig(
                    checkpointer=CalibCheckpointer(restore_checkpoint_path=tmp_dir)
                ),
            )

            # load state
            calib_model.load_state(checkpoint_path=tmp_dir)

            # save state
            calib_model.save_state(checkpoint_path=tmp_dir)
