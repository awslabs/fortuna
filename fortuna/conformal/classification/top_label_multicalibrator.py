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

from fortuna.conformal.multivalid.multicalibrator import Multicalibrator
from fortuna.typing import Array


class TopLabelMulticalibrator(Multicalibrator):
    def __init__(self, n_classes: int):
        """
        A multicalibration method that provides multivalid coverage guarantees. See Algorithm 15 in `Aaron Roth's notes
        <https://www.cis.upenn.edu/~aaroth/uncertainty-notes.pdf>`_.

        Parameters
        ----------
        n_classes: int
            Number of classes.
        """
        super().__init__()
        self._patch_list = []
        self.n_classes = n_classes

    def calibrate(
        self,
        targets: Array,
        groups: Optional[Array] = None,
        probs: Optional[Array] = None,
        test_groups: Optional[Array] = None,
        test_probs: Optional[Array] = None,
        tol: float = 1e-4,
        n_buckets: int = 100,
        n_rounds: int = 1000,
        eta: float = 1.0,
        **kwargs,
    ) -> Union[Dict, Tuple[Array, Dict]]:
        return super().calibrate(
            scores=self._get_scores(targets),
            groups=groups,
            values=probs,
            test_groups=test_groups,
            test_values=test_probs,
            tol=tol,
            n_buckets=n_buckets,
            n_rounds=n_rounds,
            eta=eta,
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

    def mean_squared_error(self, probs: Array, targets: Array) -> Array:
        return super().mean_squared_error(
            values=probs, scores=self._get_scores(targets)
        )

    @staticmethod
    def _get_b(
        groups: Array, values: Array, v: Array, g: Array, c: Array, n_buckets: int
    ) -> Array:
        return (
            (jnp.abs(values[:, c] - v) < 0.5 / n_buckets)
            * (values.argmax(1) == c)
            * groups[:, g]
        )

    def _patch(
        self, values: Array, patch: Array, bt: Array, ct: Array, eta: float
    ) -> Array:
        if jnp.all(~jnp.isnan(values)) and jnp.all(values.sum(1, keepdims=True) != 0.0):
            values /= values.sum(1, keepdims=True)
        return super()._patch(values=values, patch=patch, bt=bt, ct=ct, eta=eta)

    @staticmethod
    def _check_scores(scores: Array):
        pass

    @staticmethod
    def _maybe_check_values(
        values: Optional[Array], test_values: Optional[Array] = None
    ):
        if values is not None:
            if values.ndim != 2:
                raise ValueError(
                    "`probs` must be a 2-dimensional array representing the probabilities for each input "
                    "and each class."
                )
            if jnp.any(values < 0) or jnp.any(values > 1):
                raise ValueError("All elements in `values` must be within [0, 1].")
        if test_values is not None:
            if test_values.ndim != 2:
                raise ValueError(
                    "`test_probs` must be a 2-dimensional array representing the probabilities for each "
                    "input and each class."
                )
            if jnp.any(test_values < 0) or jnp.any(test_values > 1):
                raise ValueError("All elements in `test_values` must be within [0, 1].")

    def _maybe_init_values(
        self, values: Optional[Array], size: Optional[int] = None
    ) -> Array:
        if values is None:
            if size is None:
                raise ValueError(
                    "If `values` is not provided, `size` must be provided."
                )
            values = 1 / self.n_classes * jnp.ones((size, self.n_classes))
            values += 0.01 * random.normal(random.PRNGKey(0), shape=values.shape)
            values = jnp.abs(values)
            values -= 2 * jnp.maximum(0, values - 1)
            values /= values.sum(1, keepdims=True)
        else:
            values = jnp.copy(values)
        return values

    @staticmethod
    def _round_to_buckets(v: Array, buckets: Array) -> Array:
        def _fun(_v):
            return buckets[jnp.argmin(jnp.abs(_v - buckets))]

        if len(v.shape):
            return vmap(_fun)(v)
        return _fun(v)

    def _get_scores(self, targets: Array) -> Array:
        self._check_targets(targets)
        scores = []
        for i in range(self.n_classes):
            scores.append(targets == i)
        return jnp.stack(scores, axis=1)

    @staticmethod
    def _check_targets(targets: Array):
        if targets.ndim != 1:
            raise ValueError("`targets` must be a 1-dimensional array of integers.")
        if targets.dtype not in ["int32", "int64"]:
            raise ValueError("All elements in `targets` must be integers")
