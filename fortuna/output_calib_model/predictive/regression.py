from typing import List, Optional, Union

import jax.numpy as jnp
from jax._src.prng import PRNGKeyArray

from fortuna.output_calib_model.predictive.base import Predictive
from fortuna.output_calibrator.output_calib_manager.base import \
    OutputCalibManager
from fortuna.prob_output_layer.regression import RegressionProbOutputLayer
from fortuna.typing import Array


class RegressionPredictive(Predictive):
    def __init__(
        self,
        output_calib_manager: OutputCalibManager,
        prob_output_layer: RegressionProbOutputLayer,
    ):
        super().__init__(
            output_calib_manager=output_calib_manager,
            prob_output_layer=prob_output_layer,
        )

    def quantile(
        self,
        q: Union[float, Array, List],
        outputs: Array,
        n_samples: Optional[int] = 30,
        rng: Optional[PRNGKeyArray] = None,
        calibrated: bool = True,
    ) -> jnp.ndarray:
        """
        Estimate the quantile of the target variable given the output, with respect to the predictive distribution.

        Parameters
        ----------
        q: Union[float, Array, List]
            Quantile(s) to estimate.
        outputs : jnp.ndarray
            Model outputs.
        n_samples: Optional[int]
            Number of target samples to draw when computing quantiles.
        rng: Optional[PRNGKeyArray]
            A random number generator.
        calibrated : bool
            Whether the outputs should be calibrated when computing this method. If `calibrated` is set to True, the
            model must have been calibrated beforehand.

        Returns
        -------
        jnp.ndarray
            The estimated quantiles for each output.
        """
        if calibrated:
            self._check_calibrated()
            state = self.state.get()
            outputs = self.output_calib_manager.apply(
                params=state.params["output_calibrator"],
                outputs=outputs,
                mutable=state.mutable["output_calibrator"],
            )
        return self.prob_output_layer.quantile(q, outputs, n_samples, rng)

    def credible_interval(
        self,
        outputs: Array,
        n_samples: int = 30,
        error: float = 0.05,
        interval_type: str = "two-tailed",
        rng: Optional[PRNGKeyArray] = None,
        calibrated: bool = True,
    ) -> jnp.ndarray:
        """
        Estimate a credible interval of the target variable given the output, with respect to the predictive
        distribution.

        Parameters
        ----------
        outputs: Array
            Model outputs.
        n_samples: int
            Number of target samples to draw for each output.
        error: float
            The interval error. This must be a number between 0 and 1, extremes included. For example,
            `error=0.05` corresponds to a 95% level of credibility.
        interval_type: str
            The interval type. We support "two-tailed" (default), "right-tailed" and "left-tailed".
        rng : Optional[PRNGKeyArray]
            A random number generator. If not passed, this will be taken from the attributes of this class.
        calibrated : bool
            Whether the outputs should be calibrated when computing this method. If `calibrated` is set to True, the
            model must have been calibrated beforehand.

        Returns
        -------
        jnp.ndarray
            The estimated credible interval for each output.
        """
        if calibrated:
            self._check_calibrated()
            state = self.state.get()
            outputs = self.output_calib_manager.apply(
                params=state.params["output_calibrator"],
                outputs=outputs,
                mutable=state.mutable["output_calibrator"],
            )
        return self.prob_output_layer.credible_interval(
            outputs, n_samples, error, interval_type, rng
        )
