import jax.numpy as jnp

from fortuna.conformal.multivalid.mixins.multicalibrator import MulticalibratorMixin
from fortuna.conformal.multivalid.one_shot.base import OneShotMultivalidMethod
from fortuna.typing import Array


class OneShotMulticalibrator(MulticalibratorMixin, OneShotMultivalidMethod):
    def __init__(self, seed: int = 0):
        super().__init__(seed=seed)

    def _get_patch(
        self, v: Array, c: Array, scores: Array, values: Array, min_prob_b: float
    ) -> Array:
        return self._compute_expectation(
            v=v, c=c, scores=scores, values=values, min_prob_b=min_prob_b
        )

    def _compute_expectation(
        self, v: Array, c: Array, scores: Array, values: Array, min_prob_b: float
    ):
        b = self._get_b(values=values, v=v, c=c, n_buckets=self.n_buckets)
        filtered_scores = scores * b
        prob_b = jnp.mean(b)
        mean = jnp.where(prob_b > min_prob_b, jnp.mean(filtered_scores) / prob_b, v)
        return mean
