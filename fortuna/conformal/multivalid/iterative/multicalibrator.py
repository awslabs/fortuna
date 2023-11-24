from __future__ import annotations

from typing import (
    Optional,
    Tuple,
)

import jax.numpy as jnp

from fortuna.conformal.multivalid.iterative.base import IterativeMultivalidMethod
from fortuna.conformal.multivalid.mixins.multicalibrator import MulticalibratorMixin
from fortuna.typing import Array


class Multicalibrator(MulticalibratorMixin, IterativeMultivalidMethod):
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
        tau: Array,
        scores: Array,
        groups: Array,
        values: Array,
        buckets: Array,
        **kwargs,
    ):
        patch, b, prob_b = self.__calibration_error(
            v=v,
            g=g,
            c=c,
            b=None,
            tau=tau,
            scores=scores,
            groups=groups,
            values=values,
            buckets=buckets,
        )
        return prob_b * patch**2, b

    def __calibration_error(
        self,
        v: Array,
        g: Array,
        c: Array,
        b: Optional[Array],
        tau: Optional[Array],
        scores: Array,
        groups: Array,
        values: Array,
        buckets: Array,
    ) -> Tuple[Array, Array, Array]:
        ey, b, prob_b = self._compute_expectation(
            v=v,
            g=g,
            c=c,
            b=b,
            tau=tau,
            scores=scores,
            groups=groups,
            values=values,
            n_buckets=len(buckets),
        )
        patch = jnp.where(prob_b > 0, ey - self._get_mean_values(values, b, c), 0.0)
        return patch, b, prob_b

    def _compute_expectation(
        self,
        v: Array,
        g: Array,
        c: Array,
        tau: Optional[Array],
        scores: Array,
        groups: Array,
        values: Array,
        n_buckets: int,
        b: Optional[Array] = None,
    ):
        if b is None:
            b = self._get_b(
                groups=groups,
                values=values,
                v=v,
                g=g,
                c=c,
                tau=tau,
                n_buckets=n_buckets,
            )
        filtered_scores = scores * b
        prob_b = jnp.mean(b)
        mean = jnp.mean(filtered_scores) / prob_b
        return mean, b, prob_b

    def _get_patch(
        self,
        v: Array,
        g: Array,
        c: Array,
        b: Optional[Array],
        tau: Optional[Array],
        scores: Array,
        groups: Array,
        values: Array,
        buckets: Array,
        patch_type: str,
        **kwargs,
    ) -> Array:
        if patch_type == "additive":
            return self.__calibration_error(
                v=v,
                g=g,
                c=c,
                b=b,
                tau=tau,
                scores=scores,
                groups=groups,
                values=values,
                buckets=buckets,
            )[0]
        if patch_type == "multiplicative":
            ey, b, prob_b = self._compute_expectation(
                v=v,
                g=g,
                c=c,
                b=b,
                tau=tau,
                scores=scores,
                groups=groups,
                values=values,
                n_buckets=len(buckets),
            )
            return ey / self._get_mean_values(values, b, c)

    @staticmethod
    def _get_mean_values(values: Array, b: Array, c: Array):
        if values.ndim == 1:
            return jnp.mean(values * b) / jnp.mean(b)
        return jnp.mean(values[:, c] * b) / jnp.mean(b)
