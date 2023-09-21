import logging
from typing import (
    Optional,
    Union,
)

from jax import vmap
import jax.numpy as jnp

from fortuna.typing import Array


class MaxCoverageFixedPrecisionBinaryClassificationCalibrator:
    def __init__(self):
        """
        A base iterative multivalid method.

        Parameters
        ----------
        seed: int
            Random seed.
        """
        self._patches = dict()

    def calibrate(
        self,
        targets: Array,
        probs: Array,
        true_positive_precision_threshold: float,
        false_negative_precision_threshold: float,
        test_probs: Optional[Array] = None,
        n_taus: int = 100,
        margin: float = 0.
    ) -> Union[None, Array]:
        if false_negative_precision_threshold <= 0.5 or true_positive_precision_threshold <= 0.5:
            raise ValueError("Both `false_negative_precision_threshold` and"
                             " `true_positive_precision_threshold` must be greater than 0.5.")
        probs = jnp.copy(probs)
        targets = jnp.copy(targets)

        def _true_positive_objective_fn(tau: Array):
            calib_probs = jnp.clip((1 + (tau - 1) * (probs > 0.5)) * probs, a_max=1)
            b_pos_prec = calib_probs >= true_positive_precision_threshold
            prob_b_pos_prec = jnp.mean(b_pos_prec)
            pos_prec = jnp.mean(targets * b_pos_prec) / prob_b_pos_prec
            pos_cond = pos_prec >= true_positive_precision_threshold + margin
            return prob_b_pos_prec * pos_cond

        def _false_negative_objective_fn(tau: Array):
            calib_probs = (1 + (tau - 1) * (probs < 0.5)) * probs
            b_neg_prec = calib_probs <= 1 - false_negative_precision_threshold
            prob_b_neg_prec = jnp.mean(b_neg_prec)
            neg_prec = jnp.mean((1 - targets) * b_neg_prec) / prob_b_neg_prec
            neg_cond = neg_prec >= false_negative_precision_threshold + margin
            return prob_b_neg_prec * neg_cond

        taus_pos = jnp.linspace(1, 2 * true_positive_precision_threshold, n_taus)
        taus_neg = jnp.linspace(2 * (1 - false_negative_precision_threshold), 1, n_taus)[::-1]

        values_pos = vmap(_true_positive_objective_fn)(taus_pos)

        msg = "The {} could not be satisfied. Please consider improving the classifier or decreasing the threshold."

        if jnp.max(values_pos) == 0:
            logging.warning(msg.format("`true_positive_precision_threshold`"))
        values_neg = vmap(_false_negative_objective_fn)(taus_neg)
        if jnp.max(values_neg) == 0:
            logging.warning(msg.format("`false_negative_precision_threshold`"))

        self._patches["tau_pos"] = taus_pos[jnp.argmax(values_pos)]
        self._patches["tau_neg"] = taus_neg[jnp.argmax(values_neg)]

        if test_probs is not None:
            return self.apply_patches(test_probs)

    def apply_patches(
        self,
        probs: Array
    ) -> Array:
        if not len(self._patches):
            logging.warning("No patches available.")
            return probs

        probs = jnp.copy(probs)
        probs = probs.at[probs > 0.5].set(jnp.clip(self._patches["tau_pos"] * probs[probs > 0.5], a_max=1))
        probs = probs.at[probs < 0.5].set(self._patches["tau_neg"] * probs[probs < 0.5])
        return probs

    @staticmethod
    def true_positive_precision(probs: Array, targets: Array, threshold: float):
        b = probs >= threshold
        prob_b = jnp.mean(b)
        return jnp.mean(targets * b) / prob_b

    @staticmethod
    def false_negative_precision(probs: Array, targets: Array, threshold: float):
        b = probs <= 1 - threshold
        prob_b = jnp.mean(b)
        return jnp.mean((1 - targets) * b) / prob_b

    @staticmethod
    def true_positive_coverage(probs: Array, threshold: float):
        return jnp.mean(probs >= threshold)

    @staticmethod
    def false_negative_coverage(probs: Array, threshold: float):
        return jnp.mean(probs <= threshold)

    @property
    def patches(self):
        return self._patches
