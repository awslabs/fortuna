import abc
from typing import (
    Any,
    List,
    Optional,
    Tuple,
    Union,
)

import jax.numpy as jnp
from jax._src.prng import PRNGKeyArray

from fortuna.output_calibrator.output_calib_manager.base import OutputCalibManager
from fortuna.prob_output_layer.base import ProbOutputLayer
from fortuna.typing import (
    Array,
    CalibMutable,
    CalibParams,
)
from fortuna.utils.random import WithRNG


class Predictive(WithRNG, abc.ABC):
    def __init__(
        self,
        output_calib_manager: OutputCalibManager,
        prob_output_layer: ProbOutputLayer,
    ):
        r"""
        Abstract predictive distribution. It characterizes the distribution of the target variable given the
        calibrated outputs. It can be see as :math:`p(y|\omega)`, where :math:`y` is a target variable and
        :math:`\omega` a calibrated output.
        """
        self.output_calib_manager = output_calib_manager
        self.prob_output_layer = prob_output_layer
        self.state = None

    def log_prob(
        self, outputs: Array, targets: Array, calibrated: bool = True, **kwargs
    ) -> jnp.ndarray:
        """
        Evaluate the log-probability density function (a.k.a. log-pdf) given the outputs and target data.

        Parameters
        ----------
        outputs : Array
            Calibrated outputs.
        targets : Array
            Target data points.
        calibrated : bool
            Whether the outputs should be calibrated when computing this method. If `calibrated` is set to True, the
            model must have been calibrated beforehand.

        Returns
        -------
        jnp.ndarray
            An evaluation of the log-pdf for each data point.
        """
        if calibrated:
            self._check_calibrated()
            state = self.state.get()
            outputs = self.output_calib_manager.apply(
                params=state.params["output_calibrator"],
                outputs=outputs,
                mutable=state.mutable["output_calibrator"],
            )
        return self.prob_output_layer.log_prob(outputs, targets, **kwargs)

    def sample(
        self,
        n_samples: int,
        outputs: Array,
        rng: Optional[PRNGKeyArray] = None,
        calibrated: bool = True,
        **kwargs,
    ) -> jnp.ndarray:
        """
        Sample target variables for each outputs.

        Parameters
        ----------
        n_samples: int
            The number of target samples to draw for each of the outputs.
        outputs : Array
            Calibrated outputs.
        rng : Optional[PRNGKeyArray]
            A random number generator. If not passed, this will be taken from the attributes of this class.
        calibrated : bool
            Whether the outputs should be calibrated when computing this method. If `calibrated` is set to True, the
            model must have been calibrated beforehand.

        Returns
        -------
        jnp.ndarray
            Samples of the target variable for each output.
        """
        if calibrated:
            self._check_calibrated()
            state = self.state.get()
            outputs = self.output_calib_manager.apply(
                params=state.params["output_calibrator"],
                outputs=outputs,
                mutable=state.mutable["output_calibrator"],
            )
        return self.prob_output_layer.sample(n_samples, outputs, rng, **kwargs)

    def mean(self, outputs: Array, calibrated: bool = True, **kwargs) -> jnp.ndarray:
        """
        Estimate the mean of the target variable given the output, with respect to the predictive distribution.

        Parameters
        ----------
        outputs : Array
            Model outputs.
        calibrated : bool
            Whether the outputs should be calibrated when computing this method. If `calibrated` is set to True, the
            model must have been calibrated beforehand.

        Returns
        -------
        jnp.ndarray
            The estimated mean for each output.
        """
        if calibrated:
            self._check_calibrated()
            state = self.state.get()
            outputs = self.output_calib_manager.apply(
                params=state.params["output_calibrator"],
                outputs=outputs,
                mutable=state.mutable["output_calibrator"],
            )
        return self.prob_output_layer.mean(outputs, **kwargs)

    def mode(self, outputs: Array, calibrated: bool = True, **kwargs) -> jnp.ndarray:
        """
        Estimate the mode of the target variable given the output, with respect to the predictive distribution.

        Parameters
        ----------
        outputs : Array
            Model outputs.
        calibrated : bool
            Whether the outputs should be calibrated when computing this method. If `calibrated` is set to True, the
            model must have been calibrated beforehand.

        Returns
        -------
        jnp.ndarray
            The estimated mode for each output.
        """
        if calibrated:
            self._check_calibrated()
            state = self.state.get()
            outputs = self.output_calib_manager.apply(
                params=state.params["output_calibrator"],
                outputs=outputs,
                mutable=state.mutable["output_calibrator"],
            )
        return self.prob_output_layer.mode(outputs, **kwargs)

    def variance(
        self, outputs: jnp.ndarray, calibrated: bool = True, **kwargs
    ) -> jnp.ndarray:
        """
        Estimate the variance of the target variable given the output, with respect to the predictive distribution.

        Parameters
        ----------
        outputs : jnp.ndarray
            Model outputs.
        calibrated : bool
            Whether the outputs should be calibrated when computing this method. If `calibrated` is set to True, the
            model must have been calibrated beforehand.

        Returns
        -------
        jnp.ndarray
            The estimated variance for each output.
        """
        if calibrated:
            self._check_calibrated()
            state = self.state.get()
            outputs = self.output_calib_manager.apply(
                params=state.params["output_calibrator"],
                outputs=outputs,
                mutable=state.mutable["output_calibrator"],
            )
        return self.prob_output_layer.variance(outputs, **kwargs)

    def std(
        self,
        outputs: jnp.ndarray,
        variances: Optional[jnp.ndarray] = None,
        calibrated: bool = True,
    ) -> jnp.ndarray:
        """
        Estimate the standard deviation of the target variable given the output, with respect to the predictive
        distribution.

        Parameters
        ----------
        outputs : jnp.ndarray
            Model outputs.
        variances: Optional[jnp.ndarray]
            Variance for each output.
        calibrated : bool
            Whether the outputs should be calibrated when computing this method. If `calibrated` is set to True, the
            model must have been calibrated beforehand.

        Returns
        -------
        jnp.ndarray
            The estimated standard deviation for each output.
        """
        if calibrated:
            self._check_calibrated()
            state = self.state.get()
            outputs = self.output_calib_manager.apply(
                params=state.params["output_calibrator"],
                outputs=outputs,
                mutable=state.mutable["output_calibrator"],
            )
        return self.prob_output_layer.std(outputs, variances=variances)

    def entropy(
        self, outputs: jnp.ndarray, calibrated: bool = True, **kwargs
    ) -> jnp.ndarray:
        """
        Estimate the entropy of the target variable given the output, with respect to the predictive distribution.

        Parameters
        ----------
        outputs : jnp.ndarray
            Model outputs.
        calibrated : bool
            Whether the outputs should be calibrated when computing this method. If `calibrated` is set to True, the
            model must have been calibrated beforehand.

        Returns
        -------
        jnp.ndarray
            The estimated mean for each output.
        """
        if calibrated:
            self._check_calibrated()
            state = self.state.get()
            outputs = self.output_calib_manager.apply(
                params=state.params["output_calibrator"],
                outputs=outputs,
                mutable=state.mutable["output_calibrator"],
            )
        return self.prob_output_layer.entropy(outputs, **kwargs)

    def _check_calibrated(self) -> None:
        """
        Check that the model has been calibrated beforehand.
        """
        if self.state is None:
            raise ValueError(
                "No calibration state was found. The model must be calibrated beforehand."
            )

    def _log_joint_prob(
        self,
        params: CalibParams,
        targets: Array,
        outputs: Array,
        mutable: Optional[CalibMutable] = None,
        rng: Optional[PRNGKeyArray] = None,
        return_aux: Optional[List[str]] = None,
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Any]]:
        if return_aux is None:
            return_aux = []
        supported_aux = ["outputs", "mutable"]
        unsupported_aux = [s for s in return_aux if s not in supported_aux]
        if sum(unsupported_aux) > 0:
            raise AttributeError(
                """The auxiliary objects {} is unknown. Please make sure that all elements of `return_aux`
                            belong to the following list: {}""".format(
                    unsupported_aux, supported_aux
                )
            )
        aux = dict()
        outs = self.output_calib_manager.apply(
            params=params["output_calibrator"],
            outputs=outputs,
            mutable=mutable["output_calibrator"],
            rng=rng,
            calib="mutable" in return_aux,
        )
        if (
            mutable is not None
            and mutable["output_calibrator"] is not None
            and "mutable" in return_aux
        ):
            outputs, aux["mutable"] = outs
            aux["mutable"] = dict(output_calibrator=aux["mutable"])
        else:
            outputs = outs
            if "mutable" in return_aux:
                aux["mutable"] = dict(output_calibrator=None)
        log_joint_prob = self.prob_output_layer.log_prob(outputs, targets).sum()

        if len(return_aux) == 0:
            return log_joint_prob
        else:
            if "outputs" in return_aux:
                aux["outputs"] = outputs
            return log_joint_prob, aux
