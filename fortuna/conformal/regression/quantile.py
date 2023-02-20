from typing import Optional

import jax.numpy as jnp

from fortuna.typing import Array
from fortuna.conformal.regression.base import ConformalRegressor


class QuantileConformalRegressor(ConformalRegressor):
    def score(
        self,
        val_lower_bounds: Array,
        val_upper_bounds: Array,
        val_targets: Array,
    ) -> jnp.ndarray:
        """
        Compute score function.

        Parameters
        ----------
        val_lower_bounds: Array
            Interval lower bounds computed on a validation set.
        val_upper_bounds: Array
            Interval upper bounds computed on a validation set.
        val_targets: Array
            A two-dimensional array of validation target variables.

        Returns
        -------
        jnp.ndarray
            The conformal scores.
        """
        if val_lower_bounds.shape != val_upper_bounds.shape:
            raise ValueError(
                """The shapes of `val_lower_bounds` and `val_upper_bounds` must be the same, but shapes {} and {} 
                were given, respectively.""".format(
                    val_lower_bounds.shape, val_upper_bounds.shape
                )
            )
        if val_lower_bounds.ndim != 1:
            raise ValueError(
                "`val_lower_bounds` and `val_upper_bounds` must be one-dimensional arrays."
            )
        if val_targets.shape[1] != 1:
            raise ValueError(
                """The second dimension of `val_targets` must have only one component."""
            )
        val_targets = val_targets.squeeze(1)
        return jnp.maximum(
            val_lower_bounds - val_targets, val_targets - val_upper_bounds
        )

    def quantile(
        self,
        val_lower_bounds: Array,
        val_upper_bounds: Array,
        val_targets: Array,
        error: float,
        scores: Optional[Array] = None,
    ) -> Array:
        """
        Compute a quantile of the scores.

        Parameters
        ----------
        val_lower_bounds: Array
            Interval lower bounds computed on a validation set.
        val_upper_bounds: Array
            Interval upper bounds computed on a validation set.
        val_targets: Array
            A two-dimensional array of validation target variables.
        error: float
            Coverage error. This must be a scalar between 0 and 1, extremes included. This should correspond to the
            coverage error for which `val_lower_bounds`, `val_upper_bounds`, `test_lower_bounds` and
            `test_upper_bounds` were computed.
        scores: Optional[float]
            Conformal scores. This should be the output of
            :meth:`~fortuna.conformal.regression.quantile.QuantileConformalRegressor.score`.

        Returns
        -------
        float
            The conformal quantile.
        """
        if error < 0 or error > 1:
            raise ValueError("""`error` must be a scalar between 0 and 1.""")
        if scores is None:
            scores = self.score(val_lower_bounds, val_upper_bounds, val_targets)
        n = scores.shape[0]
        return jnp.quantile(scores, jnp.ceil((n + 1) * (1 - error)) / n)

    def conformal_interval(
        self,
        val_lower_bounds: Array,
        val_upper_bounds: Array,
        test_lower_bounds: Array,
        test_upper_bounds: Array,
        val_targets: Array,
        error: float,
        quantile: Optional[float] = None,
    ) -> jnp.ndarray:
        """
        Coverage interval of each of the test inputs, at the desired coverage error. This is supported only for
        scalar target variables.

        Parameters
        ----------
        val_lower_bounds: Array
            Interval lower bounds computed on a validation set.
        val_upper_bounds: Array
            Interval upper bounds computed on a validation set.
        test_lower_bounds: Array
            Interval lower bounds computed on a test set.
        test_upper_bounds: Array
            Interval upper bounds computed on a test set.
        val_targets: Array
            A two-dimensional array of validation target variables.
        error: float
            Coverage error. This must be a scalar between 0 and 1, extremes included. This should correspond to the
            coverage error for which `val_lower_bounds`, `val_upper_bounds`, `test_lower_bounds` and
            `test_upper_bounds` were computed.
        quantile: Optional[float]
            Conformal quantiles. This should be the output of
            :meth:`~fortuna.conformal.regression.quantile.QuantileConformalRegressor.quantile`.

        Returns
        -------
        jnp.ndarray
            The conformal intervals. The two components of the second axis correspond to the left and right interval
            bounds.
        """
        if val_lower_bounds.shape != val_upper_bounds.shape:
            raise ValueError(
                f"""The shapes of `val_lower_bounds` and `val_upper_bounds` must be the same, but shapes 
                {val_lower_bounds.shape} and {val_upper_bounds.shape} were found, respectively."""
            )

        if test_lower_bounds.shape != test_upper_bounds.shape:
            raise ValueError(
                f"""The shapes of `test_lower_bounds` and `test_upper_bounds` must be the same, but shapes 
                {test_lower_bounds.shape} and {test_upper_bounds.shape} were found, respectively."""
            )

        if val_lower_bounds.ndim not in [1, 2]:
            raise ValueError(
                "`val_lower_bounds` and `val_upper_bounds` must be one- or two-dimensional arrays. If "
                "two-dimensional, the second dimension must be 1."
            )
        if test_lower_bounds.ndim not in [1, 2]:
            raise ValueError(
                "`test_lower_bounds` and `test_upper_bounds` must be one- or two-dimensional arrays. If "
                "two-dimensional, the second dimension must be 1."
            )
        if val_lower_bounds.ndim == 2:
            if val_lower_bounds.shape[1] != 1:
                raise ValueError(
                    f"The second dimension of `val_lower_bounds` must have only one component, but"
                    f"{val_lower_bounds.shape[1]} components were found."
                )
            else:
                val_lower_bounds = val_lower_bounds.squeeze(1)
                val_upper_bounds = val_upper_bounds.squeeze(1)
        if test_lower_bounds.ndim == 2:
            if test_lower_bounds.shape[1] != 1:
                raise ValueError(
                    f"The second dimension of `test_lower_bounds` must have only one component, but"
                    f"{test_lower_bounds.shape[1]} components were found."
                )
            else:
                test_lower_bounds = test_lower_bounds.squeeze(1)
                test_upper_bounds = test_upper_bounds.squeeze(1)

        if quantile is None:
            quantile = self.quantile(
                val_lower_bounds, val_upper_bounds, val_targets, error
            )
        lows = test_lower_bounds - quantile
        highs = test_upper_bounds + quantile
        return jnp.array(list(zip(lows, highs)))
