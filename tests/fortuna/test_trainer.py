from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Tuple,
    Union,
)
import unittest
import unittest.mock as mock

import chex
import optax
from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax.random
import flax.linen as nn
import numpy as np
from flax.core import FrozenDict
from jax import numpy as jnp
from jax._src.prng import PRNGKeyArray
import jax.random
import numpy as np
from optax._src.base import PyTree

from fortuna.model import MLP
from fortuna.prob_model.posterior.map.map_state import MAPState
from fortuna.prob_model.posterior.map.map_trainer import JittedMAPTrainer, MAPTrainer, MultiDeviceMAPTrainer
from fortuna.metric.classification import accuracy
from fortuna.training.callback import Callback
from fortuna.training.train_state import TrainState
from fortuna.training.trainer import TrainerABC
from fortuna.typing import (
    Batch,
    CalibMutable,
    CalibParams,
    Mutable,
    Params,
)


class FakeTrainState:
    apply_fn = lambda *x: x[-1]
    tx = None
    params = {}
    mutable = None
    unravel = None
    step = 0
    predict_fn = lambda *x: x[-1]


class FakeTrainer(TrainerABC):
    def training_step(
        self,
        state: TrainState,
        batch: Tuple[Union[jnp.ndarray, np.ndarray], Union[jnp.ndarray, np.ndarray]],
        log_joint_prob: Callable[[Any], Union[float, Tuple[float, dict]]],
        rng: jnp.ndarray,
        n_data: int,
        unravel: Optional[Callable[[any], PyTree]] = None,
        kwargs: FrozenDict[str, Any] = FrozenDict(),
    ) -> Tuple[TrainState, Dict[str, Any]]:
        state.step += 1
        return state, {
            "loss": 4.2,
            "logging_kwargs": None,
        }

    def training_loss_step(
        self,
        fun: Callable[[Any], Union[float, Tuple[float, dict]]],
        params: Params,
        batch: Batch,
        mutable: Mutable,
        rng: PRNGKeyArray,
        n_data: int,
        unravel: Optional[Callable[[any], PyTree]] = None,
        calib_params: Optional[CalibParams] = None,
        calib_mutable: Optional[CalibMutable] = None,
        kwargs: FrozenDict[str, Any] = FrozenDict(),
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        pass

    def validation_step(
        self,
        state: TrainState,
        batch: Tuple[Union[jnp.ndarray, np.ndarray], Union[jnp.ndarray, np.ndarray]],
        log_joint_prob: Callable[[Any], Union[float, Tuple[float, dict]]],
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
        trainer._global_training_step = 10
        state = FakeTrainState()
        batch = [[1, 2, 3], [0, 0, 1]]
        with unittest.mock.patch.object(
            trainer, "save_checkpoint"
        ) as msc, unittest.mock.patch.object(trainer, "_callback_loop") as cl:
            with self.assertRaises(KeyError):
                trainer.training_step_end(1, state, {}, batch, ())
        msc.assert_called_once_with(state, "tmp_dir", keep=3)
        cl.assert_not_called()

        with unittest.mock.patch.object(
            trainer, "save_checkpoint"
        ) as msc, unittest.mock.patch.object(trainer, "_callback_loop") as cl:
            with self.assertRaises(KeyError):
                trainer.training_step_end(1, state, {"loss": 1}, batch, ())
        msc.assert_called_once_with(state, "tmp_dir", keep=3)
        cl.assert_not_called()

        with unittest.mock.patch.object(
            trainer, "save_checkpoint"
        ) as msc, unittest.mock.patch.object(trainer, "_callback_loop") as cl:
            with self.assertRaises(KeyError):
                trainer.training_step_end(
                    1, state, {"loss": 1, "logging_kwargs": None}, batch, ()
                )
        msc.assert_called_once_with(state, "tmp_dir", keep=3)
        cl.assert_not_called()

    def test_training_step_end_ok_no_training_metrics_computation(self):
        trainer = FakeTrainer(
            predict_fn=lambda *args, **kwargs: args[0],
            disable_training_metrics_computation=True,
            save_checkpoint_dir="tmp_dir",
            save_every_n_steps=1,
            keep_top_n_checkpoints=3,
        )
        trainer._global_training_step = 10
        state = FakeTrainState()
        batch = [[1, 2, 3], [0, 0, 1]]

        with unittest.mock.patch.object(
            trainer, "save_checkpoint"
        ) as msc, unittest.mock.patch.object(
            trainer, "_callback_loop", side_effect=trainer._callback_loop
        ) as cl:
            observed_state, training_losses_and_metrics = trainer.training_step_end(
                1,
                state,
                {"loss": 1, "logging_kwargs": None},
                batch,
                (),
            )
        msc.assert_called_once_with(state, "tmp_dir", keep=3)
        cl.assert_called_once_with(state, None, "training_step_end")
        self.assertEqual(training_losses_and_metrics, {"loss": 1})
        self.assertEqual(state, observed_state)

        with unittest.mock.patch.object(
            trainer, "save_checkpoint"
        ) as msc, unittest.mock.patch.object(
            trainer, "_callback_loop", side_effect=trainer._callback_loop
        ) as cl:
            observed_state, training_losses_and_metrics = trainer.training_step_end(
                1,
                state,
                {"loss": 1, "logging_kwargs": {"metric1:": 0.1, "metrics2": 0.2}},
                batch,
                (),
            )
        msc.assert_called_once_with(state, "tmp_dir", keep=3)
        cl.assert_called_once_with(state, None, "training_step_end")
        self.assertEqual(
            training_losses_and_metrics, {"loss": 1, "metric1:": 0.1, "metrics2": 0.2}
        )
        self.assertEqual(state, observed_state)

        trainer = FakeTrainer(
            predict_fn=lambda *args, **kwargs: args[0],
            disable_training_metrics_computation=False,
            save_checkpoint_dir="tmp_dir",
            save_every_n_steps=1,
            keep_top_n_checkpoints=3,
        )
        trainer._global_training_step = 10

        with unittest.mock.patch.object(
            trainer, "save_checkpoint"
        ) as msc, unittest.mock.patch.object(
            trainer, "_callback_loop", side_effect=trainer._callback_loop
        ) as cl:
            observed_state, training_losses_and_metrics = trainer.training_step_end(
                1, state, {"loss": 1, "logging_kwargs": None}, batch, None, []
            )
        msc.assert_called_once_with(state, "tmp_dir", keep=3)
        cl.assert_called_once_with(state, [], "training_step_end")
        self.assertEqual(training_losses_and_metrics, {"loss": 1})
        self.assertEqual(state, observed_state)

        with unittest.mock.patch.object(
            trainer, "save_checkpoint"
        ) as msc, unittest.mock.patch.object(
            trainer, "_callback_loop", side_effect=trainer._callback_loop
        ) as cl:
            observed_state, training_losses_and_metrics = trainer.training_step_end(
                1,
                state,
                {"loss": 1, "logging_kwargs": {"metric1:": 0.1, "metrics2": 0.2}},
                batch,
                None,
                [],
            )
        msc.assert_called_once_with(state, "tmp_dir", keep=3)
        cl.assert_called_once_with(state, [], "training_step_end")
        self.assertEqual(
            training_losses_and_metrics, {"loss": 1, "metric1:": 0.1, "metrics2": 0.2}
        )
        self.assertEqual(state, observed_state)

    def test_training_step_end_ok(self):
        trainer = FakeTrainer(
            predict_fn=lambda *args, **kwargs: args[0],
            disable_training_metrics_computation=False,
            save_checkpoint_dir="tmp_dir",
            save_every_n_steps=1,
            keep_top_n_checkpoints=3,
        )
        trainer._global_training_step = 10
        state = FakeTrainState()
        batch = [[1, 2, 3], [0, 0, 1]]

        def train_m1(a, b):
            return 12.0

        # no callbacks
        with unittest.mock.patch.object(
            trainer, "save_checkpoint"
        ) as msc, unittest.mock.patch.object(
            trainer, "_callback_loop", side_effect=trainer._callback_loop
        ) as cl:
            observed_state, training_losses_and_metrics = trainer.training_step_end(
                1,
                state,
                {"loss": 1, "logging_kwargs": None, "outputs": [10, 20, 30]},
                batch,
                (train_m1,),
            )
        msc.assert_called_once_with(state, "tmp_dir", keep=3)
        self.assertEqual(training_losses_and_metrics, {"loss": 1, "train_m1": 12.0})
        self.assertEqual(state, observed_state)

        # with callbacks
        callbacks = [Callback(), Callback()]
        for c in callbacks:
            c.training_epoch_start = mock.MagicMock(side_effect=c.training_epoch_start)
            c.training_epoch_end = mock.MagicMock(side_effect=c.training_epoch_end)
            c.training_step_end = mock.MagicMock(side_effect=c.training_step_end)

        with unittest.mock.patch.object(trainer, "save_checkpoint") as msc:
            observed_state, training_losses_and_metrics = trainer.training_step_end(
                1,
                state,
                {"loss": 1, "logging_kwargs": None, "outputs": [10, 20, 30]},
                batch,
                (train_m1,),
                callbacks,
            )
        msc.assert_called_once_with(state, "tmp_dir", keep=3)
        for c in callbacks:
            c.training_epoch_end.assert_not_called()
            c.training_epoch_start.assert_not_called()
            c.training_step_end.assert_called_with(state)
        self.assertEqual(training_losses_and_metrics, {"loss": 1, "train_m1": 12.0})
        self.assertEqual(state, observed_state)

    def test_training_step_end_ok_no_save(self):
        pairs = [
            (None, None, 0),
            (None, None, 99),
            (None, 0, 0),
            (None, 0, 99),
            ("tmp_dir", None, 0),
            ("tmp_dir", None, 99),
            ("tmp_dir", 0, 0),
            ("tmp_dir", 0, 99),
            (None, 1, 0),
            (None, 1, 99),
            ("tmp_dir", 1, 0),
            ("tmp_dir", 100, 99),
            ("tmp_dir", 200, 100),
        ]
        for save_checkpoint_dir, save_every_n_steps, global_step in pairs:
            trainer = FakeTrainer(
                predict_fn=lambda *args, **kwargs: args[0],
                disable_training_metrics_computation=False,
                save_checkpoint_dir=save_checkpoint_dir,
                save_every_n_steps=save_every_n_steps,
                keep_top_n_checkpoints=3,
            )
            trainer._global_training_step = global_step
            state = FakeTrainState()
            batch = [[1, 2, 3], [0, 0, 1]]

            def train_m1(a, b):
                return 12.0

            with unittest.mock.patch.object(trainer, "save_checkpoint") as msc:
                observed_state, training_losses_and_metrics = trainer.training_step_end(
                    current_epoch=1,
                    state=state,
                    aux={"loss": 1, "logging_kwargs": None, "outputs": [10, 20, 30]},
                    batch=batch,
                    metrics=(train_m1,),
                )
            msc.assert_not_called()
            self.assertEqual(training_losses_and_metrics, {"loss": 1, "train_m1": 12.0})
            self.assertEqual(state, observed_state)

    def test_training_step_end_w_save(self):
        pairs = [
            ("tmp_dir", 1, 1),
            ("tmp_dir", 99, 99),
            ("tmp_dir", 100, 500),
        ]
        for save_checkpoint_dir, save_every_n_steps, global_step in pairs:
            trainer = FakeTrainer(
                predict_fn=lambda *args, **kwargs: args[0],
                disable_training_metrics_computation=False,
                save_checkpoint_dir=save_checkpoint_dir,
                save_every_n_steps=save_every_n_steps,
                keep_top_n_checkpoints=3,
            )
            trainer._global_training_step = global_step
            state = FakeTrainState()
            batch = [[1, 2, 3], [0, 0, 1]]

            def train_m1(a, b):
                return 12.0

            with unittest.mock.patch.object(trainer, "save_checkpoint") as msc:
                observed_state, training_losses_and_metrics = trainer.training_step_end(
                    current_epoch=1,
                    state=state,
                    aux={"loss": 1, "logging_kwargs": None, "outputs": [10, 20, 30]},
                    batch=batch,
                    metrics=(train_m1,),
                )
            msc.assert_called_with(state, save_checkpoint_dir, keep=3)
            self.assertEqual(training_losses_and_metrics, {"loss": 1, "train_m1": 12.0})
            self.assertEqual(state, observed_state)

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

    def test_training_epoch_start(self):
        state = FakeTrainState()
        trainer = FakeTrainer(
            predict_fn=lambda x: x, disable_training_metrics_computation=False
        )

        # No callbacks
        with unittest.mock.patch.object(
            trainer, "_callback_loop", side_effect=lambda *args, **kwargs: args[0]
        ) as c:
            observed_state = trainer.training_epoch_start(state)
        c.assert_called_once_with(state, None, "training_epoch_start")
        self.assertEqual(state, observed_state)

        # with callbacks
        callbacks = [Callback(), Callback()]
        for c in callbacks:
            c.training_epoch_start = mock.MagicMock(side_effect=c.training_epoch_start)
            c.training_epoch_end = mock.MagicMock(side_effect=c.training_epoch_end)
            c.training_step_end = mock.MagicMock(side_effect=c.training_step_end)

        observed_state = trainer.training_epoch_start(state, callbacks)
        for c in callbacks:
            c.training_epoch_end.assert_not_called()
            c.training_epoch_start.assert_called_with(state)
            c.training_step_end.assert_not_called()
        self.assertEqual(state, observed_state)

    def test_training_epoch_end(self):
        state = FakeTrainState()
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

        # No callbacks
        with unittest.mock.patch.object(
            trainer, "_get_mean_losses_and_metrics", return_value=fake_out
        ) as m, unittest.mock.patch.object(
            trainer, "_callback_loop", side_effect=lambda *args, **kwargs: args[0]
        ) as c:
            observed_state, observed_lm = trainer.training_epoch_end(
                losses_and_metrics, state
            )
        m.assert_called_once_with(losses_and_metrics)
        c.assert_called_once_with(state, None, "training_epoch_end")
        self.assertEqual(state, observed_state)
        self.assertDictEqual(fake_out, observed_lm)

        # with callbacks
        callbacks = [Callback(), Callback()]
        for c in callbacks:
            c.training_epoch_start = mock.MagicMock(side_effect=c.training_epoch_start)
            c.training_epoch_end = mock.MagicMock(side_effect=c.training_epoch_end)
            c.training_step_end = mock.MagicMock(side_effect=c.training_step_end)

        with unittest.mock.patch.object(
            trainer, "_get_mean_losses_and_metrics", return_value=fake_out
        ) as m:
            observed_state, observed_lm = trainer.training_epoch_end(
                losses_and_metrics, state, callbacks
            )
        m.assert_called_once_with(losses_and_metrics)
        for c in callbacks:
            c.training_epoch_end.assert_called_with(state)
            c.training_epoch_start.assert_not_called()
            c.training_step_end.assert_not_called()
        self.assertEqual(state, observed_state)
        self.assertDictEqual(fake_out, observed_lm)

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
            loss_fun=lambda x: x,
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
            loss_fun=lambda x: x,
            rng=jax.random.PRNGKey(0),
            metrics=(accuracy,),
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
            loss_fun=lambda x: x,
            metrics=(accuracy,),
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

    def test_grad_acc_and_clipping(self):
        def loss_fn(params, batch, **kwargs):
            logits = model.apply(params['model'], batch[0])
            return optax.softmax_cross_entropy_with_integer_labels(logits, batch[1]).mean(), {}

        # setup
        inputs = jax.random.uniform(jax.random.PRNGKey(0), (9, 28, 28, 1))
        targets = jax.random.randint(jax.random.PRNGKey(0), (9,), minval=0, maxval=10)

        rng, _ = jax.random.split(jax.random.PRNGKey(0), 2)
        model = MLP(output_dim=10, widths=(20_000, 20_000), activations=(nn.relu, nn.relu))
        params = model.init(rng, jnp.empty((1, *inputs.shape[1:])), train=False)
        trainer = MultiDeviceMAPTrainer(predict_fn=None)
        max_grad_norm = 0.1
        gradient_accumulation_steps = 3


        # full batch update
        optimizer = optax.sgd(learning_rate=1e-4)
        state = MAPState.init(params=FrozenDict({'model': params}), optimizer=optimizer)
        new_state, _, rng = trainer.on_train_start(state, inputs, rng)
        new_state, aux = trainer.training_step(
            new_state,
            (inputs[None], targets[None]),
            loss_fn,
            rng,
            1,
             None,
            FrozenDict({"max_grad_norm": max_grad_norm})
        )
        loss = float(aux['loss'])

        # grad acc update
        optimizer = optax.sgd(learning_rate=1e-4)
        state = MAPState.init(params=FrozenDict({'model': params}), optimizer=optimizer)
        new_state_acc = state
        loss_grad_acc = 0.
        batches = [
            (inputs[start:start+gradient_accumulation_steps][None], targets[start:start+gradient_accumulation_steps][None]) for start in range(0, 9, gradient_accumulation_steps)
        ]
        rng, _ = jax.random.split(jax.random.PRNGKey(0), 2)
        new_state_acc, _, rng = trainer.on_train_start(new_state_acc, batches, rng)
        for i, batch in enumerate(batches):
            new_state_acc, aux_grad_acc = trainer.training_step(
                new_state_acc,
                batch,
                loss_fn,
                rng,
                1,
                None,
                FrozenDict({"max_grad_norm": max_grad_norm, "gradient_accumulation_steps": gradient_accumulation_steps})
            )
            loss_grad_acc += float(aux_grad_acc['loss'])
        chex.assert_tree_all_close(
            new_state.params, new_state_acc.params, atol=1e-7
        )
        self.assertAlmostEqual(loss, loss_grad_acc / (i + 1), places=6)

    def test_grad_acc_no_clipping(self):
        def loss_fn(params, batch, **kwargs):
            logits = model.apply(params['model'], batch[0])
            return optax.softmax_cross_entropy_with_integer_labels(logits, batch[1]).mean(), {}

        # setup
        inputs = jax.random.uniform(jax.random.PRNGKey(0), (9, 28, 28, 1))
        targets = jax.random.randint(jax.random.PRNGKey(0), (9,), minval=0, maxval=10)

        rng, _ = jax.random.split(jax.random.PRNGKey(0), 2)
        model = MLP(output_dim=10, widths=(10_000, 10_000), activations=(nn.relu, nn.relu))
        params = model.init(rng, jnp.empty((1, *inputs.shape[1:])), train=False)
        trainer = JittedMAPTrainer(predict_fn=None)
        gradient_accumulation_steps = 3

        # full batch update
        optimizer = optax.sgd(learning_rate=1e-4)
        state = MAPState.init(params=FrozenDict({'model': params}), optimizer=optimizer)
        new_state, aux = trainer.training_step(
            state=state,
            batch=(inputs, targets),
            rng=rng,
            loss_fun=loss_fn,
            n_data=1,
        )
        loss = float(aux['loss'])

        # grad acc update
        optimizer = optax.sgd(learning_rate=1e-4)
        state = MAPState.init(params=FrozenDict({'model': params}), optimizer=optimizer)
        new_state_acc = state
        loss_grad_acc = 0.
        batches = [
            (inputs[0:3], targets[0:3]),
            (inputs[3:6], targets[3:6]),
            (inputs[6:9], targets[6:9]),
        ]
        for batch in batches:
            new_state_acc, aux_grad_acc = trainer.training_step(
                state=new_state_acc,
                batch=batch,
                rng=rng,
                loss_fun=loss_fn,
                n_data=1,
                kwargs=FrozenDict(
                    {"gradient_accumulation_steps": gradient_accumulation_steps})
            )
            loss_grad_acc += float(aux_grad_acc['loss'])
        chex.assert_tree_all_close(
            new_state.params, new_state_acc.params, atol=1e-7
        )
        self.assertAlmostEqual(loss, loss_grad_acc / 3, places=6)
