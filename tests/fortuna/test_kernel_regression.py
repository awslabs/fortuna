import unittest

from jax import random
import jax.numpy as jnp
import numpy as np

from fortuna.kernel_regression.nadaraya_watson import NadarayaWatsonKernelRegressor


class TestKernelRegression(unittest.TestCase):
    def test_nadaraya_watson(self):
        train_x = random.normal(random.PRNGKey(0), shape=(3,))
        train_y = random.normal(random.PRNGKey(1), shape=(3,))
        eval_x = random.normal(random.PRNGKey(2), shape=(4,))

        kr = NadarayaWatsonKernelRegressor(train_inputs=train_x, train_targets=train_y)
        preds = kr.predict(inputs=eval_x)
        assert preds.shape == (4,)

        kr = NadarayaWatsonKernelRegressor(
            train_inputs=np.array(train_x), train_targets=np.array(train_y)
        )
        preds = kr.predict(inputs=np.array(eval_x))
        assert preds.shape == (4,)

        with self.assertRaises(ValueError):
            kr.predict(inputs=eval_x, bandwidth=-1)
        with self.assertRaises(ValueError):
            NadarayaWatsonKernelRegressor(
                train_inputs=train_x, train_targets=train_y[None]
            )
        with self.assertRaises(ValueError):
            NadarayaWatsonKernelRegressor(
                train_inputs=train_x[None], train_targets=train_y
            )
        with self.assertRaises(ValueError):
            NadarayaWatsonKernelRegressor(
                train_inputs=jnp.concatenate((train_x, train_y)), train_targets=train_y
            )
