from typing import (
    Dict,
    Optional,
    Tuple,
    Union,
)

import jax.numpy as jnp
from jax import random

from fortuna.conformal.multivalid.multicalibrator import Multicalibrator
from fortuna.typing import Array


class BinaryClassificationMulticalibrator(Multicalibrator):
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
        **kwargs,
    ) -> Union[Dict, Tuple[Array, Dict]]:
        return super().calibrate(
            scores=targets,
            groups=groups,
            values=probs,
            test_groups=test_groups,
            test_values=test_probs,
            tol=tol,
            n_buckets=n_buckets,
            n_rounds=n_rounds,
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
            scores=targets,
            groups=groups,
            values=probs,
        )

    def mean_squared_error(self, probs: Array, targets: Array) -> Array:
        return super().mean_squared_error(values=probs, scores=targets)

    def init_probs(self, size: int) -> Array:
        """
        Initialize probabilities.

        Parameters
        ----------
        size: int
            Number of data points.

        Returns
        -------
        Array
            A probability for each data point.
        """
        return self._maybe_init_values(values=None, size=size)

    @staticmethod
    def _check_scores(scores: Array):
        if scores.ndim != 1:
            raise ValueError("`targets` must be a 1-dimensional array of integers.")
        if set(jnp.unique(scores).tolist()) != {0, 1}:
            raise ValueError("All values in `targets` must be 0 or 1.")
        if scores.dtype not in ["int32", "int64"]:
            raise ValueError("All elements in `targets` must be integers")

    @staticmethod
    def _maybe_check_values(
        values: Optional[Array], test_values: Optional[Array] = None
    ):
        if values is not None:
            if values.ndim != 1:
                raise ValueError(
                    "`probs` must be a 1-dimensional array representing the probability that the "
                    "target variable is 1."
                )
            if jnp.any(values < 0) or jnp.any(values > 1):
                raise ValueError("All elements in `values` must be within [0, 1].")
        if test_values is not None:
            if test_values.ndim != 1:
                raise ValueError(
                    "`test_probs` must be a 1-dimensional array representing the probability that the "
                    "target variable is 1."
                )
            if jnp.any(test_values < 0) or jnp.any(test_values > 1):
                raise ValueError("All elements in `test_values` must be within [0, 1].")

    def _maybe_init_values(self, values: Optional[Array], size: Optional[int] = None):
        if values is None:
            if size is None:
                raise ValueError(
                    "If `values` is not provided, `size` must be provided."
                )
            values = 0.5 * jnp.ones(size)
            values += 0.01 * random.normal(random.PRNGKey(0), shape=values.shape)

        return jnp.copy(values)
