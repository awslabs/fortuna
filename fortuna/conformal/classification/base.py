from fortuna.typing import Array
from typing import List
import jax.numpy as jnp


class ConformalClassifier:
    """
    A base conformal classifier class.
    """
    def is_in(self, values: Array, conformal_sets: List) -> Array:
        """
        Check whether the values lie within their respective conformal sets.

        Parameters
        ----------
        values: Array
            Values to check if they lie in the respective conformal sets.
        conformal_sets: Array
            A conformal set for each input data point.

        Returns
        -------
        Array
            An array of ones or zero, indicating whether the values lie within their respective conformal sets.
        """
        return jnp.array([v in s for v, s in zip(values.tolist(), conformal_sets)])
