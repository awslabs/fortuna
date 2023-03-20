import tempfile
import unittest

from fortuna.data.loader import DataLoader
from fortuna.metric.classification import accuracy, brier_score
from fortuna.metric.regression import rmse
from fortuna.model.mlp import MLP
from fortuna.output_calibrator.classification import \
    ClassificationTemperatureScaler
from fortuna.output_calibrator.regression import RegressionTemperatureScaler
from fortuna.prob_model.calib_config import (CalibConfig, CalibMonitor,
                                             CalibOptimizer)
from fortuna.prob_model.classification import ProbClassifier
from fortuna.prob_model.fit_config import FitConfig, FitMonitor
from fortuna.prob_model.fit_config.checkpointer import FitCheckpointer
from fortuna.prob_model.fit_config.optimizer import FitOptimizer
from fortuna.prob_model.posterior.deep_ensemble.deep_ensemble_posterior import \
    DeepEnsemblePosteriorApproximator
from fortuna.prob_model.posterior.laplace.laplace_posterior import \
    LaplacePosteriorApproximator
from fortuna.prob_model.posterior.map.map_approximator import \
    MAPPosteriorApproximator
from fortuna.prob_model.posterior.normalizing_flow.advi.advi_posterior import \
    ADVIPosteriorApproximator
from fortuna.prob_model.posterior.swag.swag_posterior import \
    SWAGPosteriorApproximator
from fortuna.prob_model.prior import IsotropicGaussianPrior
from fortuna.prob_model.regression import ProbRegressor
from tests.make_data import make_array_random_data
from tests.make_model import MyModel
import numpy as np

np.random.seed(42)


def brier(dummy, p, y):
    return brier_score(p, y)


class TestApproximations(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prob_class = ProbClassifier(
            model=MyModel(2), prior=IsotropicGaussianPrior()
        )

        self.reg_input_shape = (3,)
        self.reg_output_dim = 2
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

        self.class_input_shape = (2,)
        self.class_output_dim = 2
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
        self.class_fit_config_nodir_dump = FitConfig(
            optimizer=FitOptimizer(n_epochs=3),
            monitor=FitMonitor(metrics=(accuracy,)),
            checkpointer=FitCheckpointer(dump_state=True),
        )
        self.class_fit_config_dir_nodump = lambda save_dir: FitConfig(
            optimizer=FitOptimizer(n_epochs=3),
            monitor=FitMonitor(metrics=(accuracy,)),
            checkpointer=FitCheckpointer(save_checkpoint_dir=save_dir),
        )
        self.class_fit_config_dir_dump = lambda save_dir: FitConfig(
            optimizer=FitOptimizer(n_epochs=3),
            monitor=FitMonitor(metrics=(accuracy,)),
            checkpointer=FitCheckpointer(save_checkpoint_dir=save_dir, dump_state=True),
        )
        self.class_fit_config_restore = lambda restore_dir: FitConfig(
            optimizer=FitOptimizer(n_epochs=3),
            checkpointer=FitCheckpointer(restore_checkpoint_path=restore_dir),
        )
        self.reg_fit_config_nodir_nodump = FitConfig(
            optimizer=FitOptimizer(n_epochs=3), monitor=FitMonitor(metrics=(rmse,))
        )
        self.reg_fit_config_nodir_dump = FitConfig(
            optimizer=FitOptimizer(n_epochs=3),
            monitor=FitMonitor(metrics=(rmse,)),
            checkpointer=FitCheckpointer(dump_state=True),
        )
        self.reg_fit_config_dir_nodump = lambda save_dir: FitConfig(
            optimizer=FitOptimizer(n_epochs=3),
            monitor=FitMonitor(metrics=(rmse,)),
            checkpointer=FitCheckpointer(save_checkpoint_dir=save_dir),
        )
        self.reg_fit_config_dir_dump = lambda save_dir: FitConfig(
            optimizer=FitOptimizer(n_epochs=3),
            monitor=FitMonitor(metrics=(rmse,)),
            checkpointer=FitCheckpointer(save_checkpoint_dir=save_dir, dump_state=True),
        )
        self.reg_fit_config_restore = lambda restore_dir: FitConfig(
            optimizer=FitOptimizer(n_epochs=3),
            checkpointer=FitCheckpointer(restore_checkpoint_path=restore_dir),
        )
        self.class_calib_config_nodir_nodump = CalibConfig(
            optimizer=CalibOptimizer(n_epochs=3), monitor=CalibMonitor(metrics=(brier,))
        )
        self.reg_calib_config_nodir_nodump = CalibConfig(
            optimizer=CalibOptimizer(n_epochs=3)
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
            status = prob_reg.train(
                train_data_loader=self.reg_train_data_loader,
                calib_data_loader=self.reg_val_data_loader,
                val_data_loader=self.reg_val_data_loader,
                fit_config=self.reg_fit_config_nodir_nodump,
                calib_config=self.reg_calib_config_nodir_nodump,
            )
            sample = prob_reg.posterior.sample()

            # no save dir but dump
            with self.assertRaises(ValueError):
                status = prob_reg.train(
                    train_data_loader=self.reg_train_data_loader,
                    calib_data_loader=self.reg_val_data_loader,
                    val_data_loader=self.reg_val_data_loader,
                    fit_config=self.reg_fit_config_nodir_dump,
                    calib_config=self.reg_calib_config_nodir_nodump,
                )

            # save dir, no dump
            status = prob_reg.train(
                train_data_loader=self.reg_train_data_loader,
                calib_data_loader=self.reg_val_data_loader,
                val_data_loader=self.reg_val_data_loader,
                fit_config=self.reg_fit_config_dir_nodump(tmp_dir),
                calib_config=self.reg_calib_config_nodir_nodump,
            )
            sample = prob_reg.posterior.sample()
            prob_reg.posterior.load_state(tmp_dir)

            # save dir and dump
            status = prob_reg.train(
                train_data_loader=self.reg_train_data_loader,
                calib_data_loader=self.reg_val_data_loader,
                val_data_loader=self.reg_val_data_loader,
                fit_config=self.reg_fit_config_dir_dump(tmp_dir),
                calib_config=self.reg_calib_config_nodir_nodump,
            )
            sample = prob_reg.posterior.sample()
            prob_reg.posterior.load_state(tmp_dir)

            # restore
            prob_reg.train(
                train_data_loader=self.reg_train_data_loader,
                calib_data_loader=self.reg_val_data_loader,
                val_data_loader=self.reg_val_data_loader,
                fit_config=self.reg_fit_config_restore(tmp_dir),
                calib_config=self.reg_calib_config_nodir_nodump,
            )

            # load state
            prob_reg.load_state(checkpoint_path=tmp_dir)

            # save state
            prob_reg.save_state(checkpoint_path=tmp_dir)

    def test_dryrun_class_map(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            prob_class = ProbClassifier(
                model=MyModel(self.class_output_dim),
                posterior_approximator=MAPPosteriorApproximator(),
                output_calibrator=ClassificationTemperatureScaler(),
            )
            # no save dir, no dump
            status = prob_class.train(
                train_data_loader=self.class_train_data_loader,
                calib_data_loader=self.class_val_data_loader,
                val_data_loader=self.class_val_data_loader,
                fit_config=self.class_fit_config_nodir_nodump,
                calib_config=self.class_calib_config_nodir_nodump,
            )
            sample = prob_class.posterior.sample()

            # no save dir but dump
            with self.assertRaises(ValueError):
                status = prob_class.train(
                    train_data_loader=self.class_train_data_loader,
                    calib_data_loader=self.class_val_data_loader,
                    val_data_loader=self.class_val_data_loader,
                    fit_config=self.class_fit_config_nodir_dump,
                    calib_config=self.class_calib_config_nodir_nodump,
                )

            # save dir, no dump
            status = prob_class.train(
                train_data_loader=self.class_train_data_loader,
                calib_data_loader=self.class_val_data_loader,
                val_data_loader=self.class_val_data_loader,
                fit_config=self.class_fit_config_dir_nodump(tmp_dir),
                calib_config=self.class_calib_config_nodir_nodump,
            )
            sample = prob_class.posterior.sample()
            prob_class.posterior.load_state(tmp_dir)

            # save dir and dump
            status = prob_class.train(
                train_data_loader=self.class_train_data_loader,
                calib_data_loader=self.class_val_data_loader,
                val_data_loader=self.class_val_data_loader,
                fit_config=self.class_fit_config_dir_dump(tmp_dir),
                calib_config=self.class_calib_config_nodir_nodump,
            )
            sample = prob_class.posterior.sample()
            prob_class.posterior.load_state(tmp_dir)

            # restore
            prob_class.train(
                train_data_loader=self.class_train_data_loader,
                calib_data_loader=self.class_val_data_loader,
                val_data_loader=self.class_val_data_loader,
                fit_config=self.class_fit_config_restore(tmp_dir),
                calib_config=self.class_calib_config_nodir_nodump,
            )

            # load state
            prob_class.load_state(checkpoint_path=tmp_dir)

            # save state
            prob_class.save_state(checkpoint_path=tmp_dir)

    def test_dryrun_reg_advi(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            prob_reg = ProbRegressor(
                model=MyModel(self.reg_output_dim),
                likelihood_log_variance_model=MyModel(self.reg_output_dim),
                posterior_approximator=ADVIPosteriorApproximator(),
                output_calibrator=RegressionTemperatureScaler(),
            )
            # no save dir, no dump
            status = prob_reg.train(
                train_data_loader=self.reg_train_data_loader,
                calib_data_loader=self.reg_val_data_loader,
                val_data_loader=self.reg_val_data_loader,
                fit_config=self.reg_fit_config_nodir_nodump,
                calib_config=self.reg_calib_config_nodir_nodump,
            )
            sample = prob_reg.posterior.sample()

            # no save dir but dump
            with self.assertRaises(ValueError):
                status = prob_reg.train(
                    train_data_loader=self.reg_train_data_loader,
                    calib_data_loader=self.reg_val_data_loader,
                    val_data_loader=self.reg_val_data_loader,
                    fit_config=self.reg_fit_config_nodir_dump,
                    calib_config=self.reg_calib_config_nodir_nodump,
                )

            # save dir, no dump
            status = prob_reg.train(
                train_data_loader=self.reg_train_data_loader,
                calib_data_loader=self.reg_val_data_loader,
                val_data_loader=self.reg_val_data_loader,
                fit_config=self.reg_fit_config_dir_nodump(tmp_dir),
                calib_config=self.reg_calib_config_nodir_nodump,
            )
            sample = prob_reg.posterior.sample()
            prob_reg.posterior.load_state(tmp_dir)

            # save dir and dump
            status = prob_reg.train(
                train_data_loader=self.reg_train_data_loader,
                calib_data_loader=self.reg_val_data_loader,
                val_data_loader=self.reg_val_data_loader,
                fit_config=self.reg_fit_config_dir_nodump(tmp_dir),
                calib_config=self.reg_calib_config_nodir_nodump,
            )
            sample = prob_reg.posterior.sample()
            prob_reg.posterior.load_state(tmp_dir)

            # restore from advi
            prob_reg.train(
                train_data_loader=self.reg_train_data_loader,
                calib_data_loader=self.reg_val_data_loader,
                val_data_loader=self.reg_val_data_loader,
                fit_config=self.reg_fit_config_restore(tmp_dir),
                calib_config=self.reg_calib_config_nodir_nodump,
            )

            # restore from map
            prob_reg_map = ProbRegressor(
                model=MyModel(self.reg_output_dim),
                likelihood_log_variance_model=MyModel(self.reg_output_dim),
                posterior_approximator=MAPPosteriorApproximator(),
                output_calibrator=RegressionTemperatureScaler(),
            )
            status = prob_reg_map.train(
                train_data_loader=self.reg_train_data_loader,
                calib_data_loader=self.reg_val_data_loader,
                val_data_loader=self.reg_val_data_loader,
                fit_config=self.reg_fit_config_dir_dump(tmp_dir),
                calib_config=self.reg_calib_config_nodir_nodump,
            )
            prob_reg.train(
                train_data_loader=self.reg_train_data_loader,
                calib_data_loader=self.reg_val_data_loader,
                val_data_loader=self.reg_val_data_loader,
                fit_config=self.reg_fit_config_restore(tmp_dir),
                calib_config=self.reg_calib_config_nodir_nodump,
            )

            # load state
            prob_reg.load_state(checkpoint_path=tmp_dir)

            # save state
            prob_reg.save_state(checkpoint_path=tmp_dir)

    def test_dryrun_class_advi(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            prob_class = ProbClassifier(
                model=MyModel(self.class_output_dim),
                posterior_approximator=ADVIPosteriorApproximator(),
                output_calibrator=ClassificationTemperatureScaler(),
            )
            # no save dir, no dump
            status = prob_class.train(
                train_data_loader=self.class_train_data_loader,
                calib_data_loader=self.class_val_data_loader,
                val_data_loader=self.class_val_data_loader,
                fit_config=self.class_fit_config_nodir_nodump,
                calib_config=self.class_calib_config_nodir_nodump,
            )
            sample = prob_class.posterior.sample()

            # no save dir but dump
            with self.assertRaises(ValueError):
                status = prob_class.train(
                    train_data_loader=self.class_train_data_loader,
                    calib_data_loader=self.class_val_data_loader,
                    val_data_loader=self.class_val_data_loader,
                    fit_config=self.class_fit_config_nodir_dump,
                    calib_config=self.class_calib_config_nodir_nodump,
                )

            # save dir, no dump
            status = prob_class.train(
                train_data_loader=self.class_train_data_loader,
                calib_data_loader=self.class_val_data_loader,
                val_data_loader=self.class_val_data_loader,
                fit_config=self.class_fit_config_dir_nodump(tmp_dir),
                calib_config=self.class_calib_config_nodir_nodump,
            )
            sample = prob_class.posterior.sample()
            prob_class.posterior.load_state(tmp_dir)

            # save dir and dump
            status = prob_class.train(
                train_data_loader=self.class_train_data_loader,
                calib_data_loader=self.class_val_data_loader,
                val_data_loader=self.class_val_data_loader,
                fit_config=self.class_fit_config_dir_dump(tmp_dir),
                calib_config=self.class_calib_config_nodir_nodump,
            )
            sample = prob_class.posterior.sample()
            prob_class.posterior.load_state(tmp_dir)

            # restore from advi
            status = prob_class.train(
                train_data_loader=self.class_train_data_loader,
                calib_data_loader=self.class_val_data_loader,
                val_data_loader=self.class_val_data_loader,
                fit_config=self.class_fit_config_restore(tmp_dir),
                calib_config=self.class_calib_config_nodir_nodump,
            )

            # restore from map
            prob_class_map = ProbClassifier(
                model=MyModel(self.class_output_dim),
                posterior_approximator=MAPPosteriorApproximator(),
                output_calibrator=ClassificationTemperatureScaler(),
            )
            status = prob_class_map.train(
                train_data_loader=self.class_train_data_loader,
                calib_data_loader=self.class_val_data_loader,
                val_data_loader=self.class_val_data_loader,
                fit_config=self.class_fit_config_dir_dump(tmp_dir),
                calib_config=self.class_calib_config_nodir_nodump,
            )
            status = prob_class.train(
                train_data_loader=self.class_train_data_loader,
                calib_data_loader=self.class_val_data_loader,
                val_data_loader=self.class_val_data_loader,
                fit_config=self.class_fit_config_restore(tmp_dir),
                calib_config=self.class_calib_config_nodir_nodump,
            )

            # load state
            prob_class.load_state(checkpoint_path=tmp_dir)

            # save state
            prob_class.save_state(checkpoint_path=tmp_dir)

    def test_dryrun_reg_deep_ensemble(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            prob_reg = ProbRegressor(
                model=MyModel(self.reg_output_dim),
                likelihood_log_variance_model=MyModel(self.reg_output_dim),
                posterior_approximator=DeepEnsemblePosteriorApproximator(),
                output_calibrator=RegressionTemperatureScaler(),
            )
            # no save dir, no dump
            status = prob_reg.train(
                train_data_loader=self.reg_train_data_loader,
                calib_data_loader=self.reg_val_data_loader,
                val_data_loader=self.reg_val_data_loader,
                fit_config=self.reg_fit_config_nodir_nodump,
                calib_config=self.reg_calib_config_nodir_nodump,
            )
            sample = prob_reg.posterior.sample()

            # no save dir but dump
            with self.assertRaises(ValueError):
                status = prob_reg.train(
                    train_data_loader=self.reg_train_data_loader,
                    calib_data_loader=self.reg_val_data_loader,
                    val_data_loader=self.reg_val_data_loader,
                    fit_config=self.reg_fit_config_nodir_dump,
                    calib_config=self.reg_calib_config_nodir_nodump,
                )

            # save dir, no dump
            status = prob_reg.train(
                train_data_loader=self.reg_train_data_loader,
                calib_data_loader=self.reg_val_data_loader,
                val_data_loader=self.reg_val_data_loader,
                fit_config=self.reg_fit_config_dir_nodump(tmp_dir),
                calib_config=self.reg_calib_config_nodir_nodump,
            )
            sample = prob_reg.posterior.sample()
            prob_reg.posterior.load_state(tmp_dir)

            # save dir and dump
            status = prob_reg.train(
                train_data_loader=self.reg_train_data_loader,
                calib_data_loader=self.reg_val_data_loader,
                val_data_loader=self.reg_val_data_loader,
                fit_config=self.reg_fit_config_dir_nodump(tmp_dir),
                calib_config=self.reg_calib_config_nodir_nodump,
            )
            sample = prob_reg.posterior.sample()
            prob_reg.posterior.load_state(tmp_dir)

            # restore
            status = prob_reg.train(
                train_data_loader=self.reg_train_data_loader,
                calib_data_loader=self.reg_val_data_loader,
                val_data_loader=self.reg_val_data_loader,
                fit_config=self.reg_fit_config_restore(tmp_dir),
                calib_config=self.reg_calib_config_nodir_nodump,
            )

            # load state
            prob_reg.load_state(checkpoint_path=tmp_dir)

            # save state
            prob_reg.save_state(checkpoint_path=tmp_dir)

    def test_dryrun_class_deep_ensemble(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            prob_class = ProbClassifier(
                model=MyModel(self.class_output_dim),
                posterior_approximator=DeepEnsemblePosteriorApproximator(),
                output_calibrator=ClassificationTemperatureScaler(),
            )
            # no save dir, no dump
            status = prob_class.train(
                train_data_loader=self.class_train_data_loader,
                calib_data_loader=self.class_val_data_loader,
                val_data_loader=self.class_val_data_loader,
                fit_config=self.class_fit_config_nodir_nodump,
                calib_config=self.class_calib_config_nodir_nodump,
            )
            sample = prob_class.posterior.sample()

            # no save dir but dump
            with self.assertRaises(ValueError):
                status = prob_class.train(
                    train_data_loader=self.class_train_data_loader,
                    calib_data_loader=self.class_val_data_loader,
                    val_data_loader=self.class_val_data_loader,
                    fit_config=self.class_fit_config_nodir_dump,
                    calib_config=self.class_calib_config_nodir_nodump,
                )

            # save dir, no dump
            status = prob_class.train(
                train_data_loader=self.class_train_data_loader,
                calib_data_loader=self.class_val_data_loader,
                val_data_loader=self.class_val_data_loader,
                fit_config=self.class_fit_config_dir_nodump(tmp_dir),
                calib_config=self.class_calib_config_nodir_nodump,
            )
            sample = prob_class.posterior.sample()
            prob_class.posterior.load_state(tmp_dir)

            # save dir and dump
            status = prob_class.train(
                train_data_loader=self.class_train_data_loader,
                calib_data_loader=self.class_val_data_loader,
                val_data_loader=self.class_val_data_loader,
                fit_config=self.class_fit_config_dir_dump(tmp_dir),
                calib_config=self.class_calib_config_nodir_nodump,
            )
            sample = prob_class.posterior.sample()
            prob_class.posterior.load_state(tmp_dir)

            # restore
            status = prob_class.train(
                train_data_loader=self.class_train_data_loader,
                calib_data_loader=self.class_val_data_loader,
                val_data_loader=self.class_val_data_loader,
                fit_config=self.class_fit_config_restore(tmp_dir),
                calib_config=self.class_calib_config_nodir_nodump,
            )

            # load state
            prob_class.load_state(checkpoint_path=tmp_dir)

            # save state
            prob_class.save_state(checkpoint_path=tmp_dir)

    def test_dryrun_reg_laplace(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            prob_reg = ProbRegressor(
                model=MyModel(self.reg_output_dim),
                likelihood_log_variance_model=MyModel(self.reg_output_dim),
                posterior_approximator=LaplacePosteriorApproximator(),
                output_calibrator=RegressionTemperatureScaler(),
            )
            # no save dir, no dump
            status = prob_reg.train(
                train_data_loader=self.reg_train_data_loader,
                calib_data_loader=self.reg_val_data_loader,
                val_data_loader=self.reg_val_data_loader,
                map_fit_config=self.reg_fit_config_nodir_nodump,
                fit_config=self.reg_fit_config_nodir_nodump,
                calib_config=self.reg_calib_config_nodir_nodump,
            )
            sample = prob_reg.posterior.sample()

            # no save dir but dump
            with self.assertRaises(ValueError):
                status = prob_reg.train(
                    train_data_loader=self.reg_train_data_loader,
                    calib_data_loader=self.reg_val_data_loader,
                    val_data_loader=self.reg_val_data_loader,
                    map_fit_config=self.reg_fit_config_nodir_nodump,
                    fit_config=self.reg_fit_config_nodir_dump,
                    calib_config=self.reg_calib_config_nodir_nodump,
                )

            # save dir, no dump
            status = prob_reg.train(
                train_data_loader=self.reg_train_data_loader,
                calib_data_loader=self.reg_val_data_loader,
                val_data_loader=self.reg_val_data_loader,
                map_fit_config=self.reg_fit_config_nodir_nodump,
                fit_config=self.reg_fit_config_dir_nodump(tmp_dir),
                calib_config=self.reg_calib_config_nodir_nodump,
            )
            sample = prob_reg.posterior.sample()
            prob_reg.posterior.load_state(tmp_dir)

            # save dir and dump
            status = prob_reg.train(
                train_data_loader=self.reg_train_data_loader,
                calib_data_loader=self.reg_val_data_loader,
                val_data_loader=self.reg_val_data_loader,
                map_fit_config=self.reg_fit_config_nodir_nodump,
                fit_config=self.reg_fit_config_dir_dump(tmp_dir),
                calib_config=self.reg_calib_config_nodir_nodump,
            )
            sample = prob_reg.posterior.sample()
            prob_reg.posterior.load_state(tmp_dir)

            # restore from laplace
            prob_reg.train(
                train_data_loader=self.reg_train_data_loader,
                calib_data_loader=self.reg_val_data_loader,
                val_data_loader=self.reg_val_data_loader,
                map_fit_config=self.reg_fit_config_nodir_nodump,
                fit_config=self.reg_fit_config_restore(tmp_dir),
                calib_config=self.reg_calib_config_nodir_nodump,
            )

            # restore from map
            prob_reg_map = ProbRegressor(
                model=MyModel(self.reg_output_dim),
                likelihood_log_variance_model=MyModel(self.reg_output_dim),
                posterior_approximator=MAPPosteriorApproximator(),
                output_calibrator=RegressionTemperatureScaler(),
            )
            status = prob_reg_map.train(
                train_data_loader=self.reg_train_data_loader,
                calib_data_loader=self.reg_val_data_loader,
                val_data_loader=self.reg_val_data_loader,
                map_fit_config=self.reg_fit_config_nodir_nodump,
                fit_config=self.reg_fit_config_dir_dump(tmp_dir),
                calib_config=self.reg_calib_config_nodir_nodump,
            )
            status = prob_reg.train(
                train_data_loader=self.reg_train_data_loader,
                calib_data_loader=self.reg_val_data_loader,
                val_data_loader=self.reg_val_data_loader,
                map_fit_config=self.reg_fit_config_nodir_nodump,
                fit_config=self.reg_fit_config_restore(tmp_dir),
                calib_config=self.reg_calib_config_nodir_nodump,
            )

            # load state
            prob_reg.load_state(checkpoint_path=tmp_dir)

            # save state
            prob_reg.save_state(checkpoint_path=tmp_dir)

    def test_dryrun_class_laplace(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            prob_class = ProbClassifier(
                model=MyModel(self.class_output_dim),
                posterior_approximator=LaplacePosteriorApproximator(),
                output_calibrator=ClassificationTemperatureScaler(),
            )
            # no save dir, no dump
            status = prob_class.train(
                train_data_loader=self.class_train_data_loader,
                calib_data_loader=self.class_val_data_loader,
                val_data_loader=self.class_val_data_loader,
                map_fit_config=self.class_fit_config_nodir_nodump,
                fit_config=self.class_fit_config_nodir_nodump,
                calib_config=self.class_calib_config_nodir_nodump,
            )
            sample = prob_class.posterior.sample()

            # no save dir but dump
            with self.assertRaises(ValueError):
                status = prob_class.train(
                    train_data_loader=self.class_train_data_loader,
                    calib_data_loader=self.class_val_data_loader,
                    val_data_loader=self.class_val_data_loader,
                    map_fit_config=self.class_fit_config_nodir_nodump,
                    fit_config=self.class_fit_config_nodir_dump,
                    calib_config=self.class_calib_config_nodir_nodump,
                )

            # save dir, no dump
            status = prob_class.train(
                train_data_loader=self.class_train_data_loader,
                calib_data_loader=self.class_val_data_loader,
                val_data_loader=self.class_val_data_loader,
                map_fit_config=self.class_fit_config_nodir_nodump,
                fit_config=self.class_fit_config_dir_nodump(tmp_dir),
                calib_config=self.class_calib_config_nodir_nodump,
            )
            sample = prob_class.posterior.sample()
            prob_class.posterior.load_state(tmp_dir)

            # save dir and dump
            status = prob_class.train(
                train_data_loader=self.class_train_data_loader,
                calib_data_loader=self.class_val_data_loader,
                val_data_loader=self.class_val_data_loader,
                map_fit_config=self.class_fit_config_nodir_nodump,
                fit_config=self.class_fit_config_dir_dump(tmp_dir),
                calib_config=self.class_calib_config_nodir_nodump,
            )
            sample = prob_class.posterior.sample()
            prob_class.posterior.load_state(tmp_dir)

            # restore from laplace
            status = prob_class.train(
                train_data_loader=self.class_train_data_loader,
                calib_data_loader=self.class_val_data_loader,
                val_data_loader=self.class_val_data_loader,
                map_fit_config=self.class_fit_config_nodir_nodump,
                fit_config=self.class_fit_config_restore(tmp_dir),
                calib_config=self.class_calib_config_nodir_nodump,
            )

            # restore from map
            prob_class_map = ProbClassifier(
                model=MyModel(self.class_output_dim),
                posterior_approximator=MAPPosteriorApproximator(),
                output_calibrator=ClassificationTemperatureScaler(),
            )
            status = prob_class_map.train(
                train_data_loader=self.class_train_data_loader,
                calib_data_loader=self.class_val_data_loader,
                val_data_loader=self.class_val_data_loader,
                map_fit_config=self.class_fit_config_nodir_nodump,
                fit_config=self.class_fit_config_dir_dump(tmp_dir),
                calib_config=self.class_calib_config_nodir_nodump,
            )
            status = prob_class.train(
                train_data_loader=self.class_train_data_loader,
                calib_data_loader=self.class_val_data_loader,
                val_data_loader=self.class_val_data_loader,
                map_fit_config=self.class_fit_config_nodir_nodump,
                fit_config=self.class_fit_config_restore(tmp_dir),
                calib_config=self.class_calib_config_nodir_nodump,
            )

            # load state
            prob_class.load_state(checkpoint_path=tmp_dir)

            # save state
            prob_class.save_state(checkpoint_path=tmp_dir)

    def test_dryrun_reg_swag(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            prob_reg = ProbRegressor(
                model=MyModel(self.reg_output_dim),
                likelihood_log_variance_model=MyModel(self.reg_output_dim),
                posterior_approximator=SWAGPosteriorApproximator(rank=2),
                output_calibrator=RegressionTemperatureScaler(),
            )
            # no save dir, no dump
            status = prob_reg.train(
                train_data_loader=self.reg_train_data_loader,
                calib_data_loader=self.reg_val_data_loader,
                val_data_loader=self.reg_val_data_loader,
                map_fit_config=self.reg_fit_config_nodir_nodump,
                fit_config=self.reg_fit_config_nodir_nodump,
                calib_config=self.reg_calib_config_nodir_nodump,
            )
            sample = prob_reg.posterior.sample()

            # no save dir but dump
            with self.assertRaises(ValueError):
                status = prob_reg.train(
                    train_data_loader=self.reg_train_data_loader,
                    calib_data_loader=self.reg_val_data_loader,
                    val_data_loader=self.reg_val_data_loader,
                    map_fit_config=self.reg_fit_config_nodir_nodump,
                    fit_config=self.reg_fit_config_nodir_dump,
                    calib_config=self.reg_calib_config_nodir_nodump,
                )

            # save dir, no dump
            status = prob_reg.train(
                train_data_loader=self.reg_train_data_loader,
                calib_data_loader=self.reg_val_data_loader,
                val_data_loader=self.reg_val_data_loader,
                map_fit_config=self.reg_fit_config_nodir_nodump,
                fit_config=self.reg_fit_config_dir_nodump(tmp_dir),
                calib_config=self.reg_calib_config_nodir_nodump,
            )
            sample = prob_reg.posterior.sample()
            prob_reg.posterior.load_state(tmp_dir)

            # save dir and dump
            status = prob_reg.train(
                train_data_loader=self.reg_train_data_loader,
                calib_data_loader=self.reg_val_data_loader,
                val_data_loader=self.reg_val_data_loader,
                map_fit_config=self.reg_fit_config_nodir_nodump,
                fit_config=self.reg_fit_config_dir_nodump(tmp_dir),
                calib_config=self.reg_calib_config_nodir_nodump,
            )
            sample = prob_reg.posterior.sample()
            prob_reg.posterior.load_state(tmp_dir)

            # restore from swag
            status = prob_reg.train(
                train_data_loader=self.reg_train_data_loader,
                calib_data_loader=self.reg_val_data_loader,
                val_data_loader=self.reg_val_data_loader,
                map_fit_config=self.reg_fit_config_nodir_nodump,
                fit_config=self.reg_fit_config_restore(tmp_dir),
                calib_config=self.reg_calib_config_nodir_nodump,
            )

            # restore from map
            prob_reg_map = ProbRegressor(
                model=MyModel(self.reg_output_dim),
                likelihood_log_variance_model=MyModel(self.reg_output_dim),
                posterior_approximator=MAPPosteriorApproximator(),
                output_calibrator=RegressionTemperatureScaler(),
            )
            status = prob_reg_map.train(
                train_data_loader=self.reg_train_data_loader,
                calib_data_loader=self.reg_val_data_loader,
                val_data_loader=self.reg_val_data_loader,
                map_fit_config=self.reg_fit_config_nodir_nodump,
                fit_config=self.reg_fit_config_dir_dump(tmp_dir),
                calib_config=self.reg_calib_config_nodir_nodump,
            )
            status = prob_reg.train(
                train_data_loader=self.reg_train_data_loader,
                calib_data_loader=self.reg_val_data_loader,
                val_data_loader=self.reg_val_data_loader,
                map_fit_config=self.reg_fit_config_nodir_nodump,
                fit_config=self.reg_fit_config_restore(tmp_dir),
                calib_config=self.reg_calib_config_nodir_nodump,
            )

            # load state
            prob_reg.load_state(checkpoint_path=tmp_dir)

            # save state
            prob_reg.save_state(checkpoint_path=tmp_dir)

    def test_dryrun_class_swag(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            prob_class = ProbClassifier(
                model=MyModel(self.class_output_dim),
                posterior_approximator=SWAGPosteriorApproximator(rank=2),
                output_calibrator=ClassificationTemperatureScaler(),
            )
            # no save dir, no dump
            status = prob_class.train(
                train_data_loader=self.class_train_data_loader,
                calib_data_loader=self.class_val_data_loader,
                val_data_loader=self.class_val_data_loader,
                map_fit_config=self.class_fit_config_nodir_nodump,
                fit_config=self.class_fit_config_nodir_nodump,
                calib_config=self.class_calib_config_nodir_nodump,
            )
            sample = prob_class.posterior.sample()

            # no save dir but dump
            with self.assertRaises(ValueError):
                status = prob_class.train(
                    train_data_loader=self.class_train_data_loader,
                    calib_data_loader=self.class_val_data_loader,
                    val_data_loader=self.class_val_data_loader,
                    map_fit_config=self.class_fit_config_nodir_nodump,
                    fit_config=self.class_fit_config_nodir_dump,
                    calib_config=self.class_calib_config_nodir_nodump,
                )

            # save dir, no dump
            status = prob_class.train(
                train_data_loader=self.class_train_data_loader,
                calib_data_loader=self.class_val_data_loader,
                val_data_loader=self.class_val_data_loader,
                map_fit_config=self.class_fit_config_nodir_nodump,
                fit_config=self.class_fit_config_dir_nodump(tmp_dir),
                calib_config=self.class_calib_config_nodir_nodump,
            )
            sample = prob_class.posterior.sample()
            prob_class.posterior.load_state(tmp_dir)

            # save dir and dump
            status = prob_class.train(
                train_data_loader=self.class_train_data_loader,
                calib_data_loader=self.class_val_data_loader,
                val_data_loader=self.class_val_data_loader,
                map_fit_config=self.class_fit_config_nodir_nodump,
                fit_config=self.class_fit_config_dir_dump(tmp_dir),
                calib_config=self.class_calib_config_nodir_nodump,
            )
            sample = prob_class.posterior.sample()
            prob_class.posterior.load_state(tmp_dir)

            # restore from swag
            status = prob_class.train(
                train_data_loader=self.class_train_data_loader,
                calib_data_loader=self.class_val_data_loader,
                val_data_loader=self.class_val_data_loader,
                map_fit_config=self.class_fit_config_nodir_nodump,
                fit_config=self.class_fit_config_restore(tmp_dir),
                calib_config=self.class_calib_config_nodir_nodump,
            )

            # restore from map
            prob_class_map = ProbClassifier(
                model=MyModel(self.class_output_dim),
                posterior_approximator=MAPPosteriorApproximator(),
                output_calibrator=ClassificationTemperatureScaler(),
            )
            status = prob_class_map.train(
                train_data_loader=self.class_train_data_loader,
                calib_data_loader=self.class_val_data_loader,
                val_data_loader=self.class_val_data_loader,
                map_fit_config=self.class_fit_config_nodir_nodump,
                fit_config=self.class_fit_config_dir_dump(tmp_dir),
                calib_config=self.class_calib_config_nodir_nodump,
            )
            status = prob_class.train(
                train_data_loader=self.class_train_data_loader,
                map_fit_config=self.class_fit_config_nodir_nodump,
                fit_config=self.class_fit_config_restore(tmp_dir),
                calib_config=self.class_calib_config_nodir_nodump,
            )

            # load state
            prob_class.load_state(checkpoint_path=tmp_dir)

            # save state
            prob_class.save_state(checkpoint_path=tmp_dir)
