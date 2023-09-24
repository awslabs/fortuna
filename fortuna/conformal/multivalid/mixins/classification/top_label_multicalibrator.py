from typing import Optional

from jax import vmap
import jax.numpy as jnp

from fortuna.conformal.multivalid.mixins.multicalibrator import MulticalibratorMixin
from fortuna.typing import Array


class TopLabelMulticalibratorMixin(MulticalibratorMixin):
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
        super().__init__(seed=seed)
        self._patch_list = []
        self.n_classes = n_classes

    def mean_squared_error(self, probs: Array, targets: Array) -> Array:
        return super().mean_squared_error(
            values=probs, scores=self._get_scores(targets)
        )

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

    @staticmethod
    def _maybe_normalize(values: Array):
        if jnp.all(~jnp.isnan(values)) and jnp.all(values.sum(1, keepdims=True) != 0.0):
            values /= values.sum(1, keepdims=True)
        return values
