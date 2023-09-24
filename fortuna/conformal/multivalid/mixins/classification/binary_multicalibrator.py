from typing import Optional

import jax.numpy as jnp

from fortuna.conformal.multivalid.mixins.multicalibrator import MulticalibratorMixin
from fortuna.typing import Array


class BinaryClassificationMulticalibratorMixin(MulticalibratorMixin):
    def mean_squared_error(self, probs: Array, targets: Array) -> Array:
        return super().mean_squared_error(values=probs, scores=targets)

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

    @staticmethod
    def _check_scores(scores: Array):
        if scores.ndim != 1:
            raise ValueError("`targets` must be a 1-dimensional array of integers.")
        if set(jnp.unique(scores).tolist()) != {0, 1}:
            raise ValueError("All values in `targets` must be 0 or 1.")
        if scores.dtype not in ["int32", "int64"]:
            raise ValueError("All elements in `targets` must be integers")
