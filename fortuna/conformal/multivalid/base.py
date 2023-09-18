from typing import (
    Optional,
    Union,
)

import jax.numpy as jnp

from fortuna.typing import Array


class MultivalidMethod:
    def __init__(self, seed: int = 0):
        """
        A base multivalid method.

        Parameters
        ----------
        seed: int
            Random seed.
        """
        self._seed = seed
        self._patches = None
        self._n_buckets = None

    def mean_squared_error(self, values: Array, scores: Array) -> Array:
        """
        The mean squared error between the model evaluations and the scores.
        This is supposed to decrease at every round of the algorithm.

        Parameters
        ----------
        values: Array
            The model evaluations.
        scores: Array
            The scores.

        Returns
        -------
        Array
            The mean-squared error.
        """
        return self._mean_squared_error(values, scores)

    @property
    def patches(self):
        return self._patches

    @property
    def n_buckets(self):
        return self._n_buckets

    @n_buckets.setter
    def n_buckets(self, n_buckets):
        self._n_buckets = n_buckets

    @staticmethod
    def _mean_squared_error(values: Array, scores: Array) -> Array:
        if scores.ndim == 2 and values.ndim == 1:
            scores = scores[:, 0]
        return jnp.mean((values - scores) ** 2)

    @staticmethod
    def _get_buckets(n_buckets: int):
        return jnp.linspace(0, 1, n_buckets)

    @staticmethod
    def _round_to_buckets(v: Array, buckets: Array):
        return buckets[jnp.argmin(jnp.abs(v - buckets))]

    @staticmethod
    def _maybe_check_values(
        values: Optional[Array], test_values: Optional[Array] = None
    ):
        if values is not None:
            if values.ndim != 1:
                raise ValueError("`values` must be a 1-dimensional array.")
            if values is not None and jnp.any(values < 0) or jnp.any(values > 1):
                raise ValueError("All elements in `values` must be within [0, 1].")
        if test_values is not None:
            if jnp.any(test_values < 0) or jnp.any(test_values > 1):
                raise ValueError("All elements in `test_values` must be within [0, 1].")

    @staticmethod
    def _check_scores(scores: Array):
        if scores.ndim != 1:
            raise ValueError("`scores` must be a 1-dimensional array.")
        if jnp.any(scores < 0) or jnp.any(scores > 1):
            raise ValueError("All elements in `scores` must be within [0, 1].")

    @staticmethod
    def _process_scores(scores: Array):
        scores = jnp.copy(scores)
        if scores.ndim == 1:
            scores = jnp.copy(scores[:, None])
        return scores

    @staticmethod
    def _maybe_init_min_prob_b(
        min_prob_b: Union[str, float], n_buckets: int, n_dims: int
    ):
        if min_prob_b == "auto":
            return 1 / (n_buckets * n_dims)
        return min_prob_b
