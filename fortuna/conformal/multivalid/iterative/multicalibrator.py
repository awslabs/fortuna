from __future__ import annotations

from typing import Optional

import jax.numpy as jnp

from fortuna.conformal.multivalid.iterative.base import IterativeMultivalidMethod
from fortuna.typing import Array


class Multicalibrator(IterativeMultivalidMethod):
    def __init__(self, seed: int = 0):
        """
        A multicalibration method that provides multivalid coverage guarantees. See Algorithm 15 in `Aaron Roth's notes
        <https://www.cis.upenn.edu/~aaroth/uncertainty-notes.pdf>`_.

        Parameters
        ----------
        seed: int
            Random seed.
        """
        super().__init__(seed=seed)
        self._patch_list = []

    def _calibration_error(
        self,
        v: Array,
        g: Array,
        c: Array,
        scores: Array,
        groups: Array,
        values: Array,
        n_buckets: int,
        **kwargs,
    ):
        expectation_error, b, prob_b = self._compute_expectation_error(
            v=v,
            g=g,
            c=c,
            scores=scores,
            groups=groups,
            values=values,
            n_buckets=n_buckets,
            return_prob_b=True,
        )
        return prob_b * expectation_error, b

    def _compute_expectation_error(
        self,
        v: Array,
        g: Array,
        c: Array,
        scores: Array,
        groups: Array,
        values: Array,
        n_buckets: int,
        return_prob_b: bool = False,
        b: Optional[Array] = None,
    ):
        mean = self._compute_expectation(
            v=v,
            g=g,
            c=c,
            scores=scores,
            groups=groups,
            values=values,
            n_buckets=n_buckets,
            return_prob_b=return_prob_b,
            b=b,
        )
        if return_prob_b:
            mean, b, prob_b = mean
            return (v - mean) ** 2, b, prob_b
        mean, b = mean
        return (v - mean) ** 2, b

    def _compute_expectation(
        self,
        v: Array,
        g: Array,
        c: Array,
        scores: Array,
        groups: Array,
        values: Array,
        n_buckets: int,
        return_prob_b: bool = False,
        b: Optional[Array] = None,
    ):
        if b is None:
            b = self._get_b(
                groups=groups, values=values, v=v, g=g, c=c, n_buckets=n_buckets
            )
        filtered_scores = scores * b
        prob_b = jnp.mean(b)
        mean = jnp.where(prob_b > 0, jnp.mean(filtered_scores) / prob_b, 0.0)

        if return_prob_b:
            return mean, b, prob_b
        return mean, b

    def _get_patch(
        self,
        vt: Array,
        gt: Array,
        ct: Array,
        bt: Array,
        scores: Array,
        groups: Array,
        values: Array,
        buckets: Array,
        **kwargs,
    ) -> Array:
        patch, bt = self._compute_expectation(
            v=vt,
            g=gt,
            c=ct,
            b=bt,
            scores=scores,
            groups=groups,
            values=values,
            n_buckets=len(buckets),
        )
        return self._round_to_buckets(patch, buckets)
