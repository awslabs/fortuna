import tempfile
import unittest

from fortuna.data.loader import DataLoader
from fortuna.metric.classification import brier_score
from fortuna.model.mlp import MLP
from fortuna.calibration.calib_model import CalibClassifier, CalibRegressor, Config, Checkpointer, Optimizer, Monitor
from tests.make_data import make_array_random_data
from tests.make_model import MyModel
from fortuna.prob_model import ProbRegressor, ProbClassifier, FitConfig, FitCheckpointer, FitOptimizer, SWAGPosteriorApproximator
import numpy as np
import jax.numpy as jnp

np.random.seed(42)


def brier(dummy, p, y):
    return brier_score(p, y)


def scaled_mse(m, v, y):
    return jnp.mean((m - y) ** 2 / v)


class TestCalibCalibrate(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        reg_calib_data = x, y
        reg_val_data = make_array_random_data(
            n_data=100,
            shape_inputs=self.reg_input_shape,
            output_dim=self.reg_output_dim,
            output_type="continuous",
        )
        reg_calib_data = [
            (reg_calib_data[0][i : i + bs], reg_calib_data[1][i : i + bs])
            for i in range(0, len(reg_calib_data[0]), bs)
        ]
        reg_val_data = [
            (reg_val_data[0][i : i + bs], reg_val_data[1][i : i + bs])
            for i in range(0, len(reg_val_data[0]), bs)
        ]
        self.reg_calib_data_loader = DataLoader.from_iterable(reg_calib_data)
        self.reg_val_data_loader = DataLoader.from_iterable(reg_val_data)

        self.class_input_shape = (2,)
        self.class_output_dim = 2
        class_calib_data = make_array_random_data(
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
        class_calib_data = [
            (class_calib_data[0][i : i + bs], class_calib_data[1][i : i + bs])
            for i in range(0, len(class_calib_data[0]), bs)
        ]
        class_val_data = [
            (class_val_data[0][i : i + bs], class_val_data[1][i : i + bs])
            for i in range(0, len(class_val_data[0]), bs)
        ]
        self.class_calib_data_loader = DataLoader.from_iterable(class_calib_data)
        self.class_val_data_loader = DataLoader.from_iterable(class_val_data)

        self.class_config_nodir_nodump = Config(
            optimizer=Optimizer(n_epochs=3), monitor=Monitor(metrics=(brier,))
        )
        self.class_config_nodir_dump = Config(
            optimizer=Optimizer(n_epochs=3),
            monitor=Monitor(metrics=(brier,)),
            checkpointer=Checkpointer(dump_state=True),
        )
        self.class_config_dir_nodump = lambda save_dir: Config(
            optimizer=Optimizer(n_epochs=3),
            monitor=Monitor(metrics=(brier,)),
            checkpointer=Checkpointer(save_checkpoint_dir=save_dir),
        )
        self.class_config_dir_dump = lambda save_dir: Config(
            optimizer=Optimizer(n_epochs=3),
            monitor=Monitor(metrics=(brier,)),
            checkpointer=Checkpointer(save_checkpoint_dir=save_dir, dump_state=True),
        )
        self.class_config_restore = lambda restore_dir: Config(
            optimizer=Optimizer(n_epochs=3),
            checkpointer=Checkpointer(restore_checkpoint_path=restore_dir),
        )
        self.reg_config_nodir_nodump = Config(
            optimizer=Optimizer(n_epochs=3), monitor=Monitor(metrics=(scaled_mse,))
        )
        self.reg_config_nodir_dump = Config(
            optimizer=Optimizer(n_epochs=3),
            monitor=Monitor(metrics=(scaled_mse,)),
            checkpointer=Checkpointer(dump_state=True),
        )
        self.reg_config_dir_nodump = lambda save_dir: Config(
            optimizer=Optimizer(n_epochs=3),
            monitor=Monitor(metrics=(scaled_mse,)),
            checkpointer=Checkpointer(save_checkpoint_dir=save_dir),
        )
        self.reg_config_dir_dump = lambda save_dir: Config(
            optimizer=Optimizer(n_epochs=3),
            monitor=Monitor(metrics=(scaled_mse,)),
            checkpointer=Checkpointer(save_checkpoint_dir=save_dir, dump_state=True),
        )
        self.reg_config_restore = lambda restore_dir: Config(
            optimizer=Optimizer(n_epochs=3),
            checkpointer=Checkpointer(restore_checkpoint_path=restore_dir),
        )

    def test_dryrun_reg(self):
        model = MLP(self.reg_output_dim)
        lik_model = MyModel(self.reg_output_dim)

        with tempfile.TemporaryDirectory() as tmp_dir:
            prob_reg = ProbRegressor(
                model=model,
                likelihood_log_variance_model=lik_model,
                posterior_approximator=SWAGPosteriorApproximator(rank=2)
            )

            status = prob_reg.train(
                train_data_loader=self.reg_calib_data_loader,
                map_fit_config=FitConfig(
                    optimizer=FitOptimizer(n_epochs=2),
                    checkpointer=FitCheckpointer(save_checkpoint_dir=tmp_dir, dump_state=True)
                )
            )

            calib_reg = CalibRegressor(
                model=model,
                likelihood_log_variance_model=lik_model,
            )

            # no save dir, no dump
            status = calib_reg.calibrate(
                calib_data_loader=self.reg_calib_data_loader,
                val_data_loader=self.reg_val_data_loader,
                config=self.reg_config_nodir_nodump,
            )

            # no save dir but dump
            with self.assertRaises(ValueError):
                status = calib_reg.calibrate(
                    calib_data_loader=self.reg_calib_data_loader,
                    val_data_loader=self.reg_val_data_loader,
                    config=self.reg_config_nodir_dump,
                )

            # save dir, no dump
            status = calib_reg.calibrate(
                calib_data_loader=self.reg_calib_data_loader,
                val_data_loader=self.reg_val_data_loader,
                config=self.reg_config_dir_nodump(tmp_dir),
            )

            # save dir and dump
            status = calib_reg.calibrate(
                calib_data_loader=self.reg_calib_data_loader,
                val_data_loader=self.reg_val_data_loader,
                config=self.reg_config_dir_dump(tmp_dir),
            )
            calib_reg.load_state(tmp_dir)

            # restore
            status = calib_reg.calibrate(
                calib_data_loader=self.reg_calib_data_loader,
                val_data_loader=self.reg_val_data_loader,
                config=self.reg_config_restore(tmp_dir),
            )

            # load state
            calib_reg.load_state(checkpoint_path=tmp_dir)

            # save state
            calib_reg.save_state(checkpoint_path=tmp_dir)

    def test_dryrun_class(self):
        model = MLP(self.class_output_dim)

        with tempfile.TemporaryDirectory() as tmp_dir:
            prob_class = ProbClassifier(
                model=model,
                posterior_approximator=SWAGPosteriorApproximator(rank=2)
            )
            status = prob_class.train(
                train_data_loader=self.class_calib_data_loader,
                map_fit_config=FitConfig(
                    optimizer=FitOptimizer(n_epochs=2),
                ),
                fit_config=FitConfig(
                    optimizer=FitOptimizer(n_epochs=3),
                    checkpointer=FitCheckpointer(save_checkpoint_dir=tmp_dir, dump_state=True)
                )

            )

            calib_class = CalibClassifier(
                model=model,
            )

            # no save dir, no dump
            status = calib_class.calibrate(
                calib_data_loader=self.class_calib_data_loader,
                val_data_loader=self.class_val_data_loader,
                config=self.class_config_nodir_nodump,
            )

            # no save dir but dump
            with self.assertRaises(ValueError):
                status = calib_class.calibrate(
                    calib_data_loader=self.class_calib_data_loader,
                    val_data_loader=self.class_val_data_loader,
                    config=self.class_config_nodir_dump,
                )

            # save dir, no dump
            status = calib_class.calibrate(
                calib_data_loader=self.class_calib_data_loader,
                val_data_loader=self.class_val_data_loader,
                config=self.class_config_dir_nodump(tmp_dir),
            )

            # save dir and dump
            status = calib_class.calibrate(
                calib_data_loader=self.class_calib_data_loader,
                val_data_loader=self.class_val_data_loader,
                config=self.class_config_dir_dump(tmp_dir),
            )
            calib_class.load_state(tmp_dir)

            # restore
            status = calib_class.calibrate(
                calib_data_loader=self.class_calib_data_loader,
                val_data_loader=self.class_val_data_loader,
                config=self.class_config_restore(tmp_dir),
            )

            # load state
            calib_class.load_state(checkpoint_path=tmp_dir)

            # save state
            calib_class.save_state(checkpoint_path=tmp_dir)

    def test_error_when_empty_data_loader(self):
        calib_class_map = CalibClassifier(
            model=MyModel(self.class_output_dim),
        )

        self.assertRaises(
            ValueError,
            lambda dl: calib_class_map.calibrate(dl),
            DataLoader.from_array_data((np.array([]), np.array([])))
        )
