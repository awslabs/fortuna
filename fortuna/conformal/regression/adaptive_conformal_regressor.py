import inspect
from typing import Optional

import jax.numpy as jnp

from fortuna.conformal.regression.base import ConformalRegressor
from fortuna.typing import Array


class AdaptiveConformalRegressor:
    def __init__(self, conformal_regressor: ConformalRegressor):
        """
        An adaptive conformal regressor class
        (see `Gibbs & Candes, 2021 <https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html>`_).
        It takes any conformal regressor and adds the functionality to update the coverage error to take into account
        distributional shifts in the data.

        Parameters
        ----------
        conformal_regressor: ConformalRegressor
            A conformal method for regression.
        """
        for s, m in inspect.getmembers(conformal_regressor):
            if not s.startswith("__"):
                setattr(self, s, m)

    def update_error(
        self,
        conformal_interval: Array,
        error: float,
        target: Array,
        target_error: float,
        gamma: float = 0.005,
        weights: Optional[Array] = None,
        were_in: Optional[Array] = None,
        return_were_in: bool = False,
    ) -> Array:
        """
        Update the coverage error based on the test target variable belonging or not to the conformal interval.

        Parameters
        ----------
        conformal_interval: List[int]
            A conformal interval for the current test target variable.
        error: float
            The current coverage error to update.
        target: Array
            The observed test target variable.
        target_error: float
            The target coverage error.
        gamma: float
            The step size for the coverage error update.
        weights: Optional[Array]
            Weights over the considered past time steps and the current one.
            This must be a one-dimensional array of increasing components between 0 and 1, summing up to 1.
        were_in: Optional[Array]
            It indicates whether the target variables of the considered past time steps fell within the respective
            conformal intervals. This must be a one-dimensional array of 1's and 0's. Its length must be the length of
            `weights` minus one, as it refers to all the past time steps but not the current one.
        return_were_in: bool
            It returns an updated `were_in`, which includes whether the current test target variable falls within its
            conformal interval.

        Returns
        -------
        Array
            The updated coverage error.
        """
        if gamma <= 0:
            raise ValueError(
                f"`gamma` must be a value greater than 0, but {gamma} was found."
            )
        if weights is not None and were_in is None:
            raise ValueError(
                "If `weights` is available, `were_in` must be available too."
            )
        if weights is None and were_in is not None:
            raise ValueError(
                "If `were_in` is available, `weights` must be available too."
            )
        if weights is not None:
            if weights.ndim > 1:
                raise ValueError(
                    "`weights` must be a one-dimensional array over the considered times in the time "
                    "series."
                )
            if (
                jnp.any(weights[:-1] > weights[1:])
                or jnp.any(weights < 0)
                or jnp.any(weights > 1)
                or not jnp.allclose(jnp.sum(weights), 1.0)
            ):
                raise ValueError(
                    "`weights` must be a vector of weights sorted in ascending order, with all elements "
                    "between 0 and 1, summing up to 1."
                )
        if were_in is not None:
            if jnp.any((were_in != 0) * (were_in != 1)):
                raise ValueError("`were_in` must be a vector of 0's and 1's.")
            if were_in.ndim != 1:
                raise ValueError(
                    "`were_in` must a be one-dimensional array over the considered times in the time "
                    "series."
                )
            if len(were_in) != len(weights) - 1:
                raise ValueError(
                    "`len(weights)-1` and `len(were_in)` must be the same. "
                    f"However, {len(weights) - 1} and {len(were_in)} were found, respectively."
                )
        is_in = self.is_in(target[None], conformal_interval[None])[0]
        if were_in is not None:
            is_in = jnp.concatenate((were_in, is_in))
            error += gamma * (target_error - jnp.dot(weights, 1 - is_in))
        else:
            error += gamma * (target_error - 1 + is_in.squeeze())
        if error > 1:
            error = 1
        if error < 0:
            error = 0

        if return_were_in:
            return float(error), is_in
        return error
