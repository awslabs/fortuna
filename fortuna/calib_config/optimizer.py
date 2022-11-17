from typing import Optional


class CalibOptimizer:
    def __init__(
        self, minimizer_kwargs: Optional[dict] = None,
    ):
        """
        An object to configure the optimization in the calibration process.

        Parameters
        ----------
        minimizer_kwargs: Optional[dict]
            Arguments to pass to the minimizer `jax.scipy.optimizer.minimize`.
        """
        self.minimizer_kwargs = minimizer_kwargs
