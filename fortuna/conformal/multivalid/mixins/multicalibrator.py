import jax.numpy as jnp

from fortuna.typing import Array


class MulticalibratorMixin:
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
        return self._loss_fn(values, scores)

    @staticmethod
    def _loss_fn(values: Array, scores: Array) -> Array:
        if scores.ndim == 2 and values.ndim == 1:
            scores = scores[:, 0]
            return jnp.mean(jnp.sum((values - scores) ** 2, axis=-1))
        return jnp.mean((values - scores) ** 2)
