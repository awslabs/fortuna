from typing import Optional

import jax.numpy as jnp

from fortuna.typing import Array
from fortuna.conformal.regression.base import ConformalRegressor


class OneDimensionalUncertaintyConformalRegressor(ConformalRegressor):
    def score(
        self,
        val_preds: Array,
        val_uncertainties: Array,
        val_targets: Array,
    ) -> jnp.ndarray:
        """
        Compute the conformal scores.

        Parameters
        ----------
        val_preds: Array
            A two-dimensional array of predictions over the validation data points.
        val_uncertainties: Array
            A two-dimensional array of uncertainty estimates (e.g. the standard deviation). The first
            dimension is over the validation inputs. The second must have only one component.
        val_targets: Array
            A two-dimensional array of validation target variables.

        Returns
        -------
        jnp.ndarray
            Scores.
        """
        if val_preds.ndim != 2 or val_preds.shape[1] != 1:
            raise ValueError(
                """`val_preds` must be a two-dimensional array. The second dimension must have only one
            component."""
            )
        if val_uncertainties.ndim != 2 or val_uncertainties.shape[1] != 1:
            raise ValueError(
                """`val_uncertainties` must be a two-dimensional array. The second dimension must have only
            one component."""
            )
        if (val_uncertainties <= 0).any():
            raise ValueError(
                """All elements in `val_uncertainties` must be strictly positive."""
            )
        if val_targets.shape[1] != 1:
            raise ValueError(
                """The second dimension of the array(s) in `val_targets` must have only one component."""
            )
        return (jnp.abs(val_targets - val_preds) / val_uncertainties).squeeze(1)

    def quantile(
        self,
        val_preds: Array,
        val_uncertainties: Array,
        val_targets: Array,
        error: float,
        scores: Optional[Array] = None,
    ) -> Array:
        """
        Compute a quantile of the scores.

        Parameters
        ----------
        val_preds: Array
            A two-dimensional array of predictions over the validation data points.
        val_uncertainties: Array
            A two-dimensional array of uncertainty estimates (e.g. the standard deviation). The first
            dimension is over the validation inputs. The second must have only one component.
        val_targets: Array
            A two-dimensional array of validation target variables.
        error: float
            Coverage error. This must be a scalar between 0 and 1, extremes included.
        scores: Optional[float]
            Conformal scores. This should be the output of
            :meth:`~fortuna.conformal.regression.onedim_uncertainty.OneDimensionalUncertaintyConformalRegressor.score`.

        Returns
        -------
        float
            The conformal quantile.
        """
        if error < 0 or error > 1:
            raise ValueError("""`error` must be a scalar between 0 and 1.""")
        if scores is None:
            scores = self.score(val_preds, val_uncertainties, val_targets)
        n = scores.shape[0]
        return jnp.quantile(scores, jnp.ceil((n + 1) * (1 - error)) / n)

    def conformal_interval(
        self,
        val_preds: Array,
        val_uncertainties: Array,
        test_preds: Array,
        test_uncertainties: Array,
        val_targets: Array,
        error: float,
        quantile: Optional[float] = None,
    ) -> jnp.ndarray:
        """
        Coverage interval of each of the test inputs, at the desired coverage error. This is supported only for
        one-dimensional target variables.

        Parameters
        ----------
        val_preds: Array
            A two-dimensional array of predictions over the validation data points.
        test_preds: Array
            A two-dimensional array of predictions over the test data points.
        val_uncertainties: Array
            A two-dimensional array of uncertainty estimates (e.g. the standard deviation). The first
            dimension is over the validation inputs. The second must have only one component.
        test_uncertainties: Array
            A two-dimensional array of uncertainty estimates (e.g. the standard deviation). The first
            dimension is over the test inputs. The second must have only one component.
        val_targets: Array
            A two-dimensional array of validation target variables.
        error: float
            Coverage error. This must be a scalar between 0 and 1, extremes included.
        quantile: Optional[float]
            Conformal quantile. This should be the output of
            :meth:`~fortuna.conformal.regression.onedim_uncertainty.OneDimensionalUncertaintyConformalRegressor.quantile`.

        Returns
        -------
        jnp.ndarray
            The conformal intervals. The two components of the second axis correspond to the left and right interval
            bounds.
        """
        if test_preds.ndim != 2 or test_preds.shape[1] != 1:
            raise ValueError(
                """`test_preds` must be a two-dimensional array. The second dimension must have only one
            component."""
            )
        if test_uncertainties.ndim != 2 or val_uncertainties.shape[1] != 1:
            raise ValueError(
                """`test_uncertainties` must be a two-dimensional array. The second dimension must have 
            only one component."""
            )
        if (test_uncertainties <= 0).any():
            raise ValueError(
                """All elements in `test_uncertainties` must be strictly positive."""
            )
        if quantile is None:
            quantile = self.quantile(val_preds, val_uncertainties, val_targets, error)
        lows = test_preds - test_uncertainties * quantile
        highs = test_preds + test_uncertainties * quantile
        return jnp.array(list(zip(lows.squeeze(1), highs.squeeze(1))))
