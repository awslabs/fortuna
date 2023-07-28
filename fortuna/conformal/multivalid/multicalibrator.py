from __future__ import annotations

import jax.numpy as jnp

from fortuna.conformal.multivalid.base import MultivalidMethod
from fortuna.typing import Array


class Multicalibrator(MultivalidMethod):
    def __init__(self):
        """
        A multicalibration method that provides multivalid coverage guarantees. See Algorithm 15 in `Aaron Roth's notes
        <https://www.cis.upenn.edu/~aaroth/uncertainty-notes.pdf>`_.
        """
        super().__init__()
        self._patch_list = []

    def _calibration_error(
        self,
        v: Array,
        g: Array,
        scores: Array,
        groups: Array,
        values: Array,
        n_buckets: int,
        **kwargs
    ):
        expectation_error, prob_b = self._compute_expectation_error(
            v=v,
            g=g,
            scores=scores,
            groups=groups,
            values=values,
            n_buckets=n_buckets,
            return_prob_b=True,
        )
        return prob_b * expectation_error

    def _compute_expectation_error(
        self,
        v: Array,
        g: Array,
        scores: Array,
        groups: Array,
        values: Array,
        n_buckets: int,
        return_prob_b: bool = False,
    ):
        if return_prob_b:
            mean, prob_b = self._compute_expectation(
                v=v,
                g=g,
                scores=scores,
                groups=groups,
                values=values,
                n_buckets=n_buckets,
                return_prob_b=return_prob_b,
            )
            return (v - mean) ** 2, prob_b
        mean = self._compute_expectation(
            v=v,
            g=g,
            scores=scores,
            groups=groups,
            values=values,
            n_buckets=n_buckets,
            return_prob_b=return_prob_b,
        )
        return (v - mean) ** 2

    def _compute_expectation(
        self,
        v: Array,
        g: Array,
        scores: Array,
        groups: Array,
        values: Array,
        n_buckets: int,
        return_prob_b: bool = False,
    ):
        b = self._get_b(groups=groups, values=values, v=v, g=g, n_buckets=n_buckets)
        filtered_scores = scores * b
        prob_b = jnp.mean(b)
        mean = jnp.where(prob_b > 0, jnp.mean(filtered_scores) / prob_b, 0.0)

        if return_prob_b:
            return mean, prob_b
        return mean

    def _get_patch(
        self,
        vt: Array,
        gt: Array,
        scores: Array,
        groups: Array,
        values: Array,
        buckets: Array,
        **kwargs
    ) -> Array:
        patch = self._compute_expectation(
            v=vt,
            g=gt,
            scores=scores,
            groups=groups,
            values=values,
            n_buckets=len(buckets),
        )
        return self._round_to_buckets(patch, buckets)
