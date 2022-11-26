import unittest
import unittest.mock as mock
from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax.random
import numpy as np
from flax.core import FrozenDict
from jax import numpy as jnp
from jax._src.prng import PRNGKeyArray
from optax._src.base import GradientTransformation, PyTree

from fortuna.prob_model.joint.state import JointState
from fortuna.training.train_state import TrainState
from fortuna.training.trainer import TrainerABC


class FakeTrainState:
    apply_fn = lambda *x: x[-1]
    tx = None
    params = {}
    mutable = None
    unravel = None
    step = 0
    predict_fn = lambda *x: x[-1]


class FakeTrainer(TrainerABC):
    def init_state(
        self,
        prob_model_state: JointState,
        predict_fn: Callable[[jnp.ndarray], jnp.ndarray],
        optimizer: GradientTransformation,
        rng: Optional[PRNGKeyArray] = None,
        **kwargs
    ) -> TrainState:
        return FakeTrainState()

    def training_step(
        self,
        state: TrainState,
        batch: Tuple[Union[jnp.ndarray, np.ndarray], Union[jnp.ndarray, np.ndarray]],
        log_prob: Callable[[Any], Union[float, Tuple[float, dict]]],
        rng: jnp.ndarray,
        n_data: int,
        unravel: Optional[Callable[[any], PyTree]] = None,
        kwargs: FrozenDict[str, Any] = FrozenDict(),
    ) -> Tuple[TrainState, Dict[str, Any]]:
        state.step += 1
        return state, {"loss": 4.2, "logging_kwargs": None,}

    def training_loss_step(
        self,
        log_prob: Callable[[Any], Union[float, Tuple[float, dict]]],
        params: Union[PyTree, jnp.ndarray, Tuple[jnp.ndarray, ...]],
        batch: Tuple[Union[jnp.ndarray, np.ndarray], Union[jnp.ndarray, np.ndarray]],
        mutable: FrozenDict[str, FrozenDict],
        rng: jnp.ndarray,
        n_data: int,
        unravel: Optional[Callable[[any], PyTree]] = None,
        kwargs: FrozenDict[str, Any] = FrozenDict(),
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        pass

    def validation_step(
        self,
        state: TrainState,
        batch: Tuple[Union[jnp.ndarray, np.ndarray], Union[jnp.ndarray, np.ndarray]],
        log_prob: Callable[[Any], Union[float, Tuple[float, dict]]],
        rng: jnp.ndarray,
        n_data: int,
        metrics: Optional[Tuple[str]] = None,
        unravel: Optional[Callable[[any], PyTree]] = None,
        kwargs: FrozenDict[str, Any] = FrozenDict(),
    ) -> Dict[str, jnp.ndarray]:
        return {"val_loss": jnp.array(0.1), "val_accuracy": jnp.array(0.5)}


class TestTrainer(unittest.TestCase):
    def test_default_init(self):
        trainer = FakeTrainer(predict_fn=lambda x: x)
        self.assertFalse(trainer.is_early_stopping_active)

    def test_training_step_end_missing_keys(self):
        trainer = FakeTrainer(
            predict_fn=lambda x: x,
            disable_training_metrics_computation=False,
            save_checkpoint_dir="tmp_dir",
            save_every_n_steps=1,
            keep_top_n_checkpoints=3,
        )
        state = FakeTrainState()
        batch = [[1, 2, 3], [0, 0, 1]]
        with unittest.mock.patch.object(trainer, "save_checkpoint") as msc:
            with self.assertRaises(KeyError):
                trainer.training_step_end(1, state, {}, batch, (), {})
        msc.assert_called_once_with(state, "tmp_dir", keep=3)

        with unittest.mock.patch.object(trainer, "save_checkpoint") as msc:
            with self.assertRaises(KeyError):
                trainer.training_step_end(1, state, {"loss": 1}, batch, (), {})
        msc.assert_called_once_with(state, "tmp_dir", keep=3)

        with unittest.mock.patch.object(trainer, "save_checkpoint") as msc:
            with self.assertRaises(KeyError):
                trainer.training_step_end(
                    1, state, {"loss": 1, "logging_kwargs": None}, batch, (), {}
                )
        msc.assert_called_once_with(state, "tmp_dir", keep=3)

    def test_training_step_end_ok_no_training_metrics_computation(self):
        trainer = FakeTrainer(
            predict_fn=lambda x: x,
            disable_training_metrics_computation=True,
            save_checkpoint_dir="tmp_dir",
            save_every_n_steps=1,
            keep_top_n_checkpoints=3,
        )
        state = FakeTrainState()
        batch = [[1, 2, 3], [0, 0, 1]]

        with unittest.mock.patch.object(trainer, "save_checkpoint") as msc:
            training_losses_and_metrics = trainer.training_step_end(
                1, state, {"loss": 1, "logging_kwargs": None}, batch, (), {}
            )
        msc.assert_called_once_with(state, "tmp_dir", keep=3)
        self.assertEqual(training_losses_and_metrics, {"loss": 1})

        with unittest.mock.patch.object(trainer, "save_checkpoint") as msc:
            training_losses_and_metrics = trainer.training_step_end(
                1,
                state,
                {"loss": 1, "logging_kwargs": {"metric1:": 0.1, "metrics2": 0.2}},
                batch,
                (),
                {},
            )
        msc.assert_called_once_with(state, "tmp_dir", keep=3)
        self.assertEqual(
            training_losses_and_metrics, {"loss": 1, "metric1:": 0.1, "metrics2": 0.2}
        )

        trainer = FakeTrainer(
            predict_fn=lambda x: x,
            disable_training_metrics_computation=False,
            save_checkpoint_dir="tmp_dir",
            save_every_n_steps=1,
            keep_top_n_checkpoints=3,
        )

        with unittest.mock.patch.object(trainer, "save_checkpoint") as msc:
            training_losses_and_metrics = trainer.training_step_end(
                1, state, {"loss": 1, "logging_kwargs": None}, batch, None, {}
            )
        msc.assert_called_once_with(state, "tmp_dir", keep=3)
        self.assertEqual(training_losses_and_metrics, {"loss": 1})

        with unittest.mock.patch.object(trainer, "save_checkpoint") as msc:
            training_losses_and_metrics = trainer.training_step_end(
                1,
                state,
                {"loss": 1, "logging_kwargs": {"metric1:": 0.1, "metrics2": 0.2}},
                batch,
                None,
                {},
            )
        msc.assert_called_once_with(state, "tmp_dir", keep=3)
        self.assertEqual(
            training_losses_and_metrics, {"loss": 1, "metric1:": 0.1, "metrics2": 0.2}
        )

    def test_training_step_end_ok(self):
        trainer = FakeTrainer(
            predict_fn=lambda x: x,
            disable_training_metrics_computation=False,
            save_checkpoint_dir="tmp_dir",
            save_every_n_steps=1,
            keep_top_n_checkpoints=3,
        )
        state = FakeTrainState()
        batch = [[1, 2, 3], [0, 0, 1]]

        def train_m1(a, b):
            return 12.0

        with unittest.mock.patch.object(trainer, "save_checkpoint") as msc:
            training_losses_and_metrics = trainer.training_step_end(
                1,
                state,
                {"loss": 1, "logging_kwargs": None, "outputs": [10, 20, 30]},
                batch,
                (train_m1,),
                {},
            )
        msc.assert_called_once_with(state, "tmp_dir", keep=3)
        self.assertEqual(training_losses_and_metrics, {"loss": 1, "train_m1": 12.0})

    def test__get_mean_losses_and_metrics_ok(self):
        trainer = FakeTrainer(
            predict_fn=lambda x: x, disable_training_metrics_computation=False
        )
        losses_and_metrics = [
            {
                "train_loss": jnp.array(0.1),
                "val_loss": jnp.array(0.2),
                "val_accuracy": jnp.array(0.1),
            },
            {
                "train_loss": jnp.array(0.05),
                "val_loss": jnp.array(0.21),
                "val_accuracy": jnp.array(0.1),
            },
            {
                "train_loss": jnp.array(0.0),
                "val_loss": jnp.array(0.22),
                "val_accuracy": jnp.array(0.1),
            },
        ]
        observed_losses_and_metrics = trainer._get_mean_losses_and_metrics(
            losses_and_metrics
        )
        expected_losses_and_metrics = {
            "train_loss": jnp.array(0.05),
            "val_accuracy": jnp.array(0.1),
            "val_loss": jnp.array(0.21),
        }
        self.assertDictEqual(observed_losses_and_metrics, expected_losses_and_metrics)

    def test__get_mean_losses_and_metrics_ko(self):
        trainer = FakeTrainer(
            predict_fn=lambda x: x, disable_training_metrics_computation=False
        )
        losses_and_metrics = [
            {
                "train_loss": jnp.array(0.1),
                "val_loss": jnp.array(0.2),
                "val_accuracy": jnp.array(0.1),
            },
            {"train_loss": jnp.array(0.05), "val_accuracy": jnp.array(0.1)},
            {
                "train_loss": jnp.array(0.0),
                "val_loss": jnp.array(0.22),
                "val_accuracy": jnp.array(0.1),
            },
        ]
        with self.assertRaises(ValueError):
            _ = trainer._get_mean_losses_and_metrics(losses_and_metrics)

    def test_training_epoch_end(self):
        trainer = FakeTrainer(
            predict_fn=lambda x: x, disable_training_metrics_computation=False
        )

        losses_and_metrics = [
            {
                "train_loss": jnp.array(0.1),
                "val_loss": jnp.array(0.2),
                "val_accuracy": jnp.array(0.1),
            },
            {
                "train_loss": jnp.array(0.0),
                "val_loss": jnp.array(0.22),
                "val_accuracy": jnp.array(0.1),
            },
        ]
        fake_out = {
            "train_loss": jnp.array(0.0),
            "val_loss": jnp.array(0.22),
            "val_accuracy": jnp.array(0.1),
        }

        with unittest.mock.patch.object(
            trainer, "_get_mean_losses_and_metrics", return_value=fake_out
        ) as m:
            observed = trainer.training_epoch_end(losses_and_metrics)
        m.assert_called_once_with(losses_and_metrics)
        self.assertDictEqual(observed, fake_out)

    def test_validation_epoch_end(self):
        trainer = FakeTrainer(
            predict_fn=lambda x: x,
            disable_training_metrics_computation=False,
            save_checkpoint_dir="tmp_dir",
            save_every_n_steps=1,
        )

        losses_and_metrics = [
            {
                "train_loss": jnp.array(0.1),
                "val_loss": jnp.array(0.2),
                "val_accuracy": jnp.array(0.1),
            },
            {
                "train_loss": jnp.array(0.0),
                "val_loss": jnp.array(0.22),
                "val_accuracy": jnp.array(0.1),
            },
        ]
        fake_out = {
            "train_loss": jnp.array(0.0),
            "val_loss": jnp.array(0.22),
            "val_accuracy": jnp.array(0.1),
        }

        with unittest.mock.patch.multiple(
            trainer,
            _get_mean_losses_and_metrics=mock.DEFAULT,
            early_stopping_update=mock.DEFAULT,
            save_checkpoint=mock.DEFAULT,
        ) as m:
            m["_get_mean_losses_and_metrics"].return_value = fake_out
            m["early_stopping_update"].return_value = False
            m["save_checkpoint"].return_value = fake_out

            observed = trainer.validation_epoch_end(losses_and_metrics, None)

        m["_get_mean_losses_and_metrics"].assert_called_once_with(losses_and_metrics)
        m["early_stopping_update"].assert_called_once_with(fake_out)
        m["save_checkpoint"].assert_not_called()
        self.assertDictEqual(observed, fake_out)

        with unittest.mock.patch.multiple(
            trainer,
            _get_mean_losses_and_metrics=mock.DEFAULT,
            early_stopping_update=mock.DEFAULT,
            save_checkpoint=mock.DEFAULT,
        ) as m:
            m["_get_mean_losses_and_metrics"].return_value = fake_out
            m["early_stopping_update"].return_value = True
            m["save_checkpoint"].return_value = fake_out

            observed = trainer.validation_epoch_end(losses_and_metrics, None)

        m["_get_mean_losses_and_metrics"].assert_called_once_with(losses_and_metrics)
        m["early_stopping_update"].assert_called_once_with(fake_out)
        m["save_checkpoint"].assert_called_once_with(None, "tmp_dir", force_save=True)
        self.assertDictEqual(observed, fake_out)

    def test_should_perform_validation(self):
        trainer = FakeTrainer(
            predict_fn=lambda x: x, disable_training_metrics_computation=False
        )
        self.assertFalse(trainer.should_perform_validation(None, 1))

        trainer.eval_every_n_epochs = 10
        self.assertFalse(trainer.should_perform_validation({}, 9))
        self.assertTrue(trainer.should_perform_validation({}, 10))

    def test__validation_loop(self):
        validation_dataloader = [
            [jnp.array([[0, 0.0, 0.0], [0, 0.0, 0]]), jnp.array([0.0, 0.0])],
            [jnp.array([[0.1, 0.0, 10], [0, 0.0, 0]]), jnp.array([1.0, 0.0])],
        ]
        trainer = FakeTrainer(
            predict_fn=lambda x: x, disable_training_metrics_computation=False
        )
        (
            observed_validation_losses_and_metrics_current_epoch,
            observed_validation_epoch_metrics_str,
        ) = trainer._validation_loop(
            state=None,
            validation_dataloader=validation_dataloader,
            validation_dataset_size=2,
            fun=lambda x: x,
            rng=jax.random.PRNGKey(0),
            metrics=("accuracy",),
            training_kwargs=FrozenDict({}),
            unravel=None,
            verbose=False,
        )
        self.assertEqual(observed_validation_epoch_metrics_str, "")
        self.assertDictEqual(
            observed_validation_losses_and_metrics_current_epoch,
            {"val_loss": jnp.array(0.1), "val_accuracy": jnp.array(0.5)},
        )

        (
            observed_validation_losses_and_metrics_current_epoch,
            observed_validation_epoch_metrics_str,
        ) = trainer._validation_loop(
            state=None,
            validation_dataloader=validation_dataloader,
            validation_dataset_size=2,
            fun=lambda x: x,
            rng=jax.random.PRNGKey(0),
            metrics=("accuracy",),
            training_kwargs=FrozenDict({}),
            unravel=None,
            verbose=True,
        )
        self.assertEqual(
            observed_validation_epoch_metrics_str, "val_accuracy: 0.5 | val_loss: 0.1"
        )
        self.assertDictEqual(
            observed_validation_losses_and_metrics_current_epoch,
            {"val_loss": jnp.array(0.1), "val_accuracy": jnp.array(0.5)},
        )

    def test__training_loop(self):
        training_dataloader = [
            [jnp.array([[0, 0.0, 0.0], [0, 0.0, 0]]), jnp.array([0.0, 0.0])],
            [jnp.array([[0.1, 0.0, 10], [0, 0.0, 0]]), jnp.array([1.0, 0.0])],
        ]
        trainer = FakeTrainer(
            predict_fn=lambda x: x, disable_training_metrics_computation=True
        )
        (
            observed_state,
            observed_train_losses_and_metrics_current_epoch,
            observed_train_epoch_metrics_str,
        ) = trainer._training_loop(
            current_epoch=1,
            fun=lambda x: x,
            metrics=("accuracy",),
            rng=jax.random.PRNGKey(0),
            state=FakeTrainState(),
            training_dataloader=training_dataloader,
            training_dataset_size=2,
            training_kwargs=FrozenDict({}),
            unravel=None,
            verbose=False,
            progress_bar=None,
        )
        self.assertEqual(observed_state.step, 2)
        self.assertEqual(observed_train_epoch_metrics_str, "")
        self.assertDictEqual(
            observed_train_losses_and_metrics_current_epoch, {"loss": jnp.array(4.2)}
        )
