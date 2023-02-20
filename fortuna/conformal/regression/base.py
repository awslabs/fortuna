from fortuna.typing import Array


class ConformalRegressor:
    """
    A base conformal regressor class.
    """
    def is_in(self, values: Array, conformal_intervals: Array) -> Array:
        """
        Check whether the values lie within their respective conformal intervals.

        Parameters
        ----------
        values: Array
            Values to check if they lie in the respective conformal intervals.
        conformal_intervals: Array
            A conformal interval for each input data point.

        Returns
        -------
        Array
            An array of ones or zero, indicating whether the values lie within their respective conformal intervals.
        """
        return (values <= conformal_intervals[:, 1]) * (values >= conformal_intervals[:, 0])
