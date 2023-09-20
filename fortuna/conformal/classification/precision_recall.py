import itertools
import logging
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from jax import (
    random,
    vmap,
)
import jax.numpy as jnp
from jax import random
import numpy as np

from fortuna.typing import Array


class MaxCoverageFixedPrecisionBinaryClassificationCalibrator:
    def __init__(self, seed: int = 0):
        """
        A base iterative multivalid method.

        Parameters
        ----------
        seed: int
            Random seed.
        """
        self._patches = []
        self._eta = None
        self._positive_threshold = None
        self._negative_threshold = None

    def calibrate(
        self,
        targets: Array,
        probs: Array,
        positive_threshold: float = 0.99,
        negative_threshold: float = 0.01,
        test_probs: Optional[Array] = None,
        min_prob_b: float = 0.1,
        n_shifts: int = 10,
        max_shift: float = 0.1
    ) -> Union[None, Array]:
        if positive_threshold <= negative_threshold:
            raise ValueError("`negative_threshold` must be strictly smaller than `positive_threshold`.")
        self._positive_threshold = positive_threshold
        self._negative_threshold = negative_threshold
        probs = jnp.copy(probs)
        targets = jnp.copy(targets)

        bs = self._get_bs(probs, positive_threshold=positive_threshold, negative_threshold=negative_threshold)

        shifts_idx = np.zeros_like(probs, dtype=int)
        for i, b in enumerate(bs):
            shifts_idx[b] = i

        def _objective_fn(s: Array):
            calib_probs = probs + s[shifts_idx]

            b_pos_prec = calib_probs >= positive_threshold
            b_neg_prec = calib_probs <= negative_threshold

            prob_b_pos_prec = jnp.mean(b_pos_prec)
            prob_b_neg_prec = jnp.mean(b_neg_prec)

            pos_prec = jnp.mean(targets[b_pos_prec] * b_pos_prec) / prob_b_pos_prec
            neg_prec = jnp.mean(targets[b_neg_prec] * b_neg_prec) / prob_b_neg_prec

            pos_cond = jnp.where(prob_b_pos_prec > min_prob_b, pos_prec >= positive_threshold, 0)
            neg_cond = jnp.where(prob_b_neg_prec > min_prob_b, neg_prec <= negative_threshold, 0)

            pos_cov = jnp.mean(calib_probs >= positive_threshold) * pos_cond
            neg_cov = jnp.mean(calib_probs <= negative_threshold) * neg_cond

            return pos_cov + neg_cov

        buckets = jnp.linspace(0, max_shift, n_shifts)
        all_shifts = jnp.array(list(itertools.combinations(buckets, len(bs))))
        self._patches = all_shifts[jnp.argmax(vmap(_objective_fn)(all_shifts))]

        if test_probs is not None:
            return self.apply_patches(test_probs)

    def apply_patches(
        self,
        probs: Array
    ) -> Array:
        if not len(self._patches):
            logging.warning("No patches available.")
            return probs

        bs = self._get_bs(
            probs=probs,
            positive_threshold=self._positive_threshold,
            negative_threshold=self._negative_threshold
        )

        probs = jnp.copy(probs)
        for b, patch in zip(bs, self._patches):
            probs = probs.at[b].set(probs[b] + patch)
        return probs

    def _get_bs(self, probs: Array, positive_threshold: float, negative_threshold: float) -> List[Array]:
        return [
            probs <= negative_threshold,
            (probs > negative_threshold) * (probs < 0.5),
            (probs >= 0.5) * (probs < positive_threshold),
            probs >= positive_threshold
        ]


if __name__ == "__main__":
    calibrator = MaxCoverageFixedPrecisionBinaryClassificationCalibrator()
    probs = jnp.abs(random.normal(random.PRNGKey(0), shape=(1000,)))
    probs /= probs.max()
    targets = random.choice(random.PRNGKey(2), 2, shape=(1000,))
    test_probs = random.normal(random.PRNGKey(1), shape=(1000,))
    test_probs /= test_probs.max()
    calib_test_probs = calibrator.calibrate(
        probs=probs,
        test_probs=test_probs,
        targets=targets,
        positive_threshold=0.99,
        negative_threshold=0.01
    )
