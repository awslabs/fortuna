from typing import (
    Optional,
    Union,
)

import jax.numpy as jnp

from fortuna.conformal.multivalid.mixins.classification.top_label_multicalibrator import (
    TopLabelMulticalibratorMixin,
)
from fortuna.conformal.multivalid.one_shot.multicalibrator import OneShotMulticalibrator
from fortuna.typing import Array


class OneShotTopLabelMulticalibrator(
    TopLabelMulticalibratorMixin, OneShotMulticalibrator
):
    def __init__(self, n_classes: int, seed: int = 0):
        super().__init__(n_classes=n_classes, seed=seed)

    def calibrate(
        self,
        targets: Array,
        probs: Optional[Array] = None,
        test_probs: Optional[Array] = None,
        n_buckets: int = 100,
        min_prob_b: Union[float, str] = "auto",
    ):
        return super().calibrate(
            scores=self._get_scores(targets),
            values=probs,
            test_values=test_probs,
            n_buckets=n_buckets,
            min_prob_b=min_prob_b,
        )

    def apply_patches(
        self,
        probs: Array,
    ) -> Array:
        return super().apply_patches(values=probs)

    @staticmethod
    def _get_b(
        values: Array,
        v: Array,
        c: Optional[Array],
        n_buckets: int,
    ) -> Array:
        return (jnp.abs(values[:, c] - v) < 0.5 / n_buckets) * (values.argmax(1) == c)
