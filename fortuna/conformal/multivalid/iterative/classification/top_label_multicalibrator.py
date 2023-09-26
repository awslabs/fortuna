from __future__ import annotations

from typing import (
    Dict,
    Optional,
    Tuple,
    Union,
)

from jax import (
    random,
    vmap,
)
import jax.numpy as jnp

from fortuna.conformal.multivalid.iterative.multicalibrator import Multicalibrator
from fortuna.conformal.multivalid.mixins.classification.top_label_multicalibrator import (
    TopLabelMulticalibratorMixin,
)
from fortuna.typing import Array


class TopLabelMulticalibrator(TopLabelMulticalibratorMixin, Multicalibrator):
    def __init__(self, n_classes: int, seed: int = 0):
        """
        A multicalibration method that provides multivalid coverage guarantees. See Algorithm 15 in `Aaron Roth's notes
        <https://www.cis.upenn.edu/~aaroth/uncertainty-notes.pdf>`_.

        Parameters
        ----------
        n_classes: int
            Number of classes.
        seed: int
            Random seed.
        """
        super().__init__(n_classes, seed=seed)

    def calibrate(
        self,
        targets: Array,
        groups: Optional[Array] = None,
        probs: Optional[Array] = None,
        test_groups: Optional[Array] = None,
        test_probs: Optional[Array] = None,
        atol: float = 1e-4,
        rtol: float = 1e-6,
        min_prob_b: Union[float, str] = "auto",
        n_buckets: int = 100,
        n_rounds: int = 1000,
        eta: float = 0.1,
        split: float = 0.8,
        bucket_types: Tuple[str, ...] = ("<=", ">="),
        **kwargs,
    ) -> Union[Dict, Tuple[Array, Dict]]:
        return super().calibrate(
            scores=self._get_scores(targets),
            groups=groups,
            values=probs,
            test_groups=test_groups,
            test_values=test_probs,
            atol=atol,
            rtol=rtol,
            min_prob_b=min_prob_b,
            n_buckets=n_buckets,
            n_rounds=n_rounds,
            eta=eta,
            split=split,
            bucket_types=bucket_types,
            **kwargs,
        )

    def apply_patches(
        self,
        groups: Optional[Array] = None,
        probs: Optional[Array] = None,
    ) -> Array:
        return super().apply_patches(groups=groups, values=probs)

    def calibration_error(
        self,
        targets: Array,
        groups: Optional[Array] = None,
        probs: Optional[Array] = None,
        n_buckets: int = 10000,
        **kwargs,
    ) -> Array:
        return super().calibration_error(
            scores=self._get_scores(targets),
            groups=groups,
            values=probs,
        )

    def _get_b(
        self,
        groups: Array,
        values: Array,
        v: Array,
        g: Array,
        c: Array,
        tau: Array,
        n_buckets: int,
    ) -> Array:
        b = super()._get_b(
            groups=groups,
            values=values[:, c],
            v=v,
            g=g,
            c=c,
            tau=tau,
            n_buckets=n_buckets,
        )
        b *= values.argmax(1) == c
        return b

    def _patch(
        self, values: Array, patch: Array, b: Array, c: Array, eta: float
    ) -> Array:
        # values = self._maybe_normalize(values)
        return super()._patch(values=values, patch=patch, b=b, c=c, eta=eta)

    def _maybe_init_values(
        self, values: Optional[Array], size: Optional[int] = None
    ) -> Array:
        if values is None:
            if size is None:
                raise ValueError(
                    "If `values` is not provided, `size` must be provided."
                )
            values = 1 / self.n_classes * jnp.ones((size, self.n_classes))
            values += 0.01 * random.normal(
                random.PRNGKey(self._seed), shape=values.shape
            )
            values = jnp.abs(values)
            values -= 2 * jnp.maximum(0, values - 1)
            values /= values.sum(1, keepdims=True)
        else:
            values = jnp.copy(values)
        return values
