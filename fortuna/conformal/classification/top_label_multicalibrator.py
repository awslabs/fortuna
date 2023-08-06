from __future__ import annotations

import jax.numpy as jnp

from fortuna.conformal.multivalid.multicalibrator import Multicalibrator
from fortuna.typing import Array
from typing import Optional, Dict, Union, Tuple


class TopLabelMulticalibrator(Multicalibrator):
    def __init__(self):
        """
        A multicalibration method that provides multivalid coverage guarantees. See Algorithm 15 in `Aaron Roth's notes
        <https://www.cis.upenn.edu/~aaroth/uncertainty-notes.pdf>`_.
        """
        super().__init__()
        self._patch_list = []

    def calibrate(
        self,
        targets: Array,
        groups: Array,
        probs: Optional[Array] = None,
        test_groups: Optional[Array] = None,
        test_probs: Optional[Array] = None,
        tol: float = 1e-4,
        n_buckets: int = 100,
        n_rounds: int = 1000,
        **kwargs,
    ) -> Union[Dict, Tuple[Array, Dict]]:
        self._check_targets(targets)
        self._check_probs(probs)

        return super().calibrate(
            scores=targets,
            groups=groups,
            values=probs,
            test_groups=test_groups,
            test_values=test_probs,
            tol=tol,
            n_buckets=n_buckets,
            n_rounds=n_rounds,
            **kwargs
        )

    def calibration_error(
        self,
        targets: Array,
        groups: Array,
        probs: Array,
        n_buckets: int = 10000,
        **kwargs,
    ) -> Array:
        self._check_targets(targets)
        self._check_probs(probs)

        return super().calibration_error(
            scores=targets,
            groups=groups,
            values=probs,
        )

    def apply_patches(
        self,
        groups: Array,
        probs: Optional[Array] = None,
    ) -> Array:
        self._check_probs(probs)

        return super().apply_patches(
            groups=groups,
            values=probs
        )

    @staticmethod
    def _get_b(
        groups: Array, values: Array, v: Array, g: Array, c: Array, n_buckets: int
    ) -> Array:
        return (jnp.abs(values[:, c] - v) < 0.5 / n_buckets) * (values.argmax(1) == c) * groups[:, g]

    @staticmethod
    def _patch(values: Array, patch: Array, bt: Array, _shift: bool = False) -> Array:
        values = super()._patch(
            values=values,
            patch=patch,
            bt=bt,
            _shift=_shift
        )
        values /= values.sum(1, keepdims=True)
        return values

    @staticmethod
    def _check_targets(targets: Array):
        if targets.ndim != 1:
            raise ValueError("`targets` must be a 1-dimensional array of integers.")
        if targets.dtype not in ["int32", "int64"]:
            raise ValueError("All elements in `targets` must be integers")

    @staticmethod
    def _check_probs(probs: Array):
        if probs.ndim != 2:
            raise ValueError("`probs` must be a 2-dimensional array or probabilities for each input data point and "
                             "each class.")
        if jnp.any(probs > 1) or jnp.any(probs < 0):
            raise ValueError("All values in `probs` must be 0 or 1.")
