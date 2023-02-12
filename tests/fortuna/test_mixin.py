import tempfile
import unittest
import unittest.mock as mock

import optax
from flax.core import FrozenDict

from fortuna.prob_model.posterior.posterior_mixin import \
    WithPosteriorCheckpointingMixin
from fortuna.prob_model.posterior.state import PosteriorState
from fortuna.training.mixin import InputValidatorMixin, WithEarlyStoppingMixin


class FakeTrainerWithCheckpointing(
    WithPosteriorCheckpointingMixin, InputValidatorMixin
):
    def __str__(self):
        return "fake"


class FakeTrainerWithEarlyStopping(WithEarlyStoppingMixin, InputValidatorMixin):
    def __str__(self):
        return "fake"


class FakeTrainState:
    apply_fn = None
    tx = None
    params = FrozenDict(dict(model=dict(params=1)), mutable=None)
    mutable = FrozenDict(dict(model=dict(params=2)), mutable=None)
    unravel = None
    step = 0


class TestCheckpointingMixins(unittest.TestCase):
    def test_init_ko(self):
        # keyword arg not recognized
        with self.assertRaises(AttributeError):
            FakeTrainerWithCheckpointing(
                save_checkpoint_dir="approximations",
                save_every_n_steps=None,
                save_top_k=1,
                NOT_A_KWARG=12,
                filepath_checkpoint_to_be_restored=None,
                use_save_checkpoint_dir_as_is=False,
            )
        # do not accept args, only kwargs
        with self.assertRaises(TypeError):
            FakeTrainerWithCheckpointing(
                123,
                save_checkpoint_dir="approximations",
                save_every_n_steps=None,
                save_top_k=1,
                filepath_checkpoint_to_be_restored=None,
                use_save_checkpoint_dir_as_is=False,
            )

    def test_save_checkpoint(self):
        state = FakeTrainState()
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = FakeTrainerWithCheckpointing()
            with unittest.mock.patch(
                "fortuna.training.mixin.checkpoints",
                return_value=mock.DEFAULT,
            ) as mc:
                trainer.save_checkpoint(state, None)
                mc.save_checkpoint.assert_not_called()

                trainer.save_checkpoint(
                    state, tmp_dir, keep=3, prefix="test_prefix_", force_save=True
                )
                mc.save_checkpoint.assert_called_with(
                    ckpt_dir=tmp_dir,
                    target=state,
                    step=state.step,
                    prefix="test_prefix_",
                    keep=3,
                    overwrite=True,
                )

    def test_restore_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = FakeTrainerWithCheckpointing()

            state = PosteriorState.init(
                params=FrozenDict(model=dict(params=2)),
                mutable=None,
                optimizer=optax.adam(1e-2),
            )
            trainer.save_checkpoint(state, tmp_dir, force_save=True)

            restored_state = trainer.restore_checkpoint(tmp_dir)
            self.assertEqual(restored_state.params["model"]["params"], 2)
            self.assertEqual(restored_state.mutable, None)

            restored_state = trainer.restore_checkpoint(
                tmp_dir, optimizer=optax.sgd(1e-1)
            )
            self.assertEqual(restored_state.params["model"]["params"], 2)
            self.assertEqual(restored_state.mutable, None)

            with unittest.mock.patch(
                "fortuna.training.mixin.checkpoints",
                return_value=mock.DEFAULT,
            ) as mc:
                mc.restore_checkpoint.return_value = FrozenDict(
                    params=dict(model=dict(params=1)),
                    encoded_name=PosteriorState.encoded_name,
                    mutable=None,
                    opt_state=dict(model=1),
                    calib_params=None,
                    calib_mutable=None,
                )
                restored_state = trainer.restore_checkpoint(
                    tmp_dir, prefix="test_prefix_"
                )
                mc.restore_checkpoint.assert_called_with(
                    ckpt_dir=tmp_dir,
                    target=None,
                    step=None,
                    prefix="test_prefix_",
                    parallel=True,
                )


class TestEarlyStoppingMixins(unittest.TestCase):
    def test_early_stopping_is_not_active(self):
        trainer = FakeTrainerWithEarlyStopping()
        with self.assertRaises(AttributeError):
            _ = trainer._early_stopping
        self.assertFalse(trainer.is_early_stopping_active)

        trainer = FakeTrainerWithEarlyStopping(
            early_stopping_monitor="val_loss",
            early_stopping_min_delta=1,
            early_stopping_patience=0,
            early_stopping_mode="min",
        )
        with self.assertRaises(AttributeError):
            _ = trainer._early_stopping
        self.assertFalse(trainer.is_early_stopping_active)

        trainer = FakeTrainerWithEarlyStopping(
            early_stopping_monitor="val_loss",
            early_stopping_min_delta=1,
            early_stopping_patience=0,
            early_stopping_mode="not_valid",
        )
        with self.assertRaises(AttributeError):
            _ = trainer._early_stopping
        self.assertFalse(trainer.is_early_stopping_active)

        trainer = FakeTrainerWithEarlyStopping(
            early_stopping_monitor="val_loss",
            early_stopping_min_delta=1,
            early_stopping_patience=0,
            early_stopping_mode="not_valid",
        )
        with self.assertRaises(AttributeError):
            _ = trainer._early_stopping
        self.assertFalse(trainer.is_early_stopping_active)

    def test_is_early_stopping_active(self):
        trainer = FakeTrainerWithEarlyStopping()
        self.assertFalse(trainer.is_early_stopping_active)

        trainer = FakeTrainerWithEarlyStopping(
            early_stopping_monitor="val_loss",
            early_stopping_min_delta=1,
            early_stopping_patience=2,
            early_stopping_mode="min",
        )
        self.assertTrue(trainer.is_early_stopping_active)

    def test_early_stopping_update_when_not_active(self):
        validation_metrics = {"metric1": 1, "metric2": 2}
        trainer = FakeTrainerWithEarlyStopping()
        self.assertIsNone(trainer.early_stopping_update(validation_metrics))

    def test_early_stopping_update_non_existing_metric(self):
        validation_metrics = {"metric1": 1, "metric2": 2}
        trainer = FakeTrainerWithEarlyStopping(
            early_stopping_monitor="val_loss",
            early_stopping_min_delta=1,
            early_stopping_patience=2,
            early_stopping_mode="min",
        )
        with self.assertRaises(KeyError):
            trainer.early_stopping_update(validation_metrics)

    def test_early_stopping_update_ok_min(self):
        validation_metrics_step1 = {"metric1": 1, "metric2": 2}
        validation_metrics_step2 = {"metric1": 0.8, "metric2": 2}
        validation_metrics_step3 = {"metric1": 0.6, "metric2": 2}
        validation_metrics_step4 = {"metric1": 1.1, "metric2": 2}
        validation_metrics_step5 = {"metric1": 1, "metric2": 2}

        trainer = FakeTrainerWithEarlyStopping(
            early_stopping_monitor="metric1",
            early_stopping_min_delta=0,
            early_stopping_patience=1,
            early_stopping_mode="min",
        )
        improved = trainer.early_stopping_update(validation_metrics_step1)
        self.assertTrue(improved)
        improved = trainer.early_stopping_update(validation_metrics_step2)
        self.assertTrue(improved)
        improved = trainer.early_stopping_update(validation_metrics_step3)
        self.assertTrue(improved)
        improved = trainer.early_stopping_update(validation_metrics_step4)
        self.assertFalse(improved)
        self.assertFalse(trainer._early_stopping.should_stop)
        improved = trainer.early_stopping_update(validation_metrics_step5)
        self.assertFalse(improved)
        self.assertTrue(trainer._early_stopping.should_stop)

        trainer = FakeTrainerWithEarlyStopping(
            early_stopping_monitor="metric1",
            early_stopping_min_delta=0,
            early_stopping_patience=2,
            early_stopping_mode="min",
        )
        improved = trainer.early_stopping_update(validation_metrics_step1)
        self.assertTrue(improved)
        improved = trainer.early_stopping_update(validation_metrics_step2)
        self.assertTrue(improved)
        improved = trainer.early_stopping_update(validation_metrics_step3)
        self.assertTrue(improved)
        improved = trainer.early_stopping_update(validation_metrics_step4)
        self.assertFalse(improved)
        self.assertFalse(trainer._early_stopping.should_stop)
        improved = trainer.early_stopping_update(validation_metrics_step5)
        self.assertFalse(improved)
        self.assertFalse(trainer._early_stopping.should_stop)
        improved = trainer.early_stopping_update(validation_metrics_step5)
        self.assertFalse(improved)
        self.assertTrue(trainer._early_stopping.should_stop)

    def test_early_stopping_update_ok_max(self):
        validation_metrics_step1 = {"metric1": 1, "metric2": 2}
        validation_metrics_step2 = {"metric1": 1.6, "metric2": 2}
        validation_metrics_step3 = {"metric1": 1.8, "metric2": 2}
        validation_metrics_step4 = {"metric1": 0.1, "metric2": 2}
        validation_metrics_step5 = {"metric1": 0.2, "metric2": 2}

        trainer = FakeTrainerWithEarlyStopping(
            early_stopping_monitor="metric1",
            early_stopping_min_delta=0,
            early_stopping_patience=2,
            early_stopping_mode="max",
        )
        improved = trainer.early_stopping_update(validation_metrics_step1)
        self.assertTrue(improved)
        improved = trainer.early_stopping_update(validation_metrics_step2)
        self.assertTrue(improved)
        improved = trainer.early_stopping_update(validation_metrics_step3)
        self.assertTrue(improved)
        improved = trainer.early_stopping_update(validation_metrics_step4)
        self.assertFalse(improved)
        self.assertFalse(trainer._early_stopping.should_stop)
        improved = trainer.early_stopping_update(validation_metrics_step5)
        self.assertFalse(improved)
        self.assertFalse(trainer._early_stopping.should_stop)
        improved = trainer.early_stopping_update(validation_metrics_step5)
        self.assertFalse(improved)
        self.assertTrue(trainer._early_stopping.should_stop)
