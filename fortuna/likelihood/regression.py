from typing import (
    Optional,
    Union,
)

import jax.numpy as jnp
import numpy as np
from jax import vmap
from jax._src.prng import PRNGKeyArray

from fortuna.data.loader import InputsLoader
from fortuna.likelihood.base import Likelihood
from fortuna.model.model_manager.regression import RegressionModelManager
from fortuna.output_calibrator.output_calib_manager.base import OutputCalibManager
from fortuna.prob_output_layer.regression import RegressionProbOutputLayer
from fortuna.typing import (
    Array,
    CalibMutable,
    CalibParams,
    Mutable,
    Params,
)


class RegressionLikelihood(Likelihood):
    def __init__(
        self,
        model_manager: RegressionModelManager,
        prob_output_layer: RegressionProbOutputLayer,
        output_calib_manager: Optional[OutputCalibManager] = None,
    ):
        """
        A regression likelihood function class. In this class, the likelihood function is additionally assumed to
        be a probability density function, i.e. positive and integrating to 1. The likelihood is formed by three
        objects applied in sequence: the model manager, the output calibrator and the probabilistic output layer. The
        model manager maps parameters and inputs to outputs. The output calibration takes outputs and returns some
        calibrated version of them. The probabilistic output layer describes the probability distribution of the
        calibrated outputs.

        Parameters
        ----------
        model_manager : ModelManager
            An model manager. This objects orchestrates the evaluation of the models.
        prob_output_layer : ProbOutputLayer
            A probabilistic output layer object. This object characterizes the probability distribution of the target
            variable given the calibrated outputs.
        output_calib_manager : Optional[OutputCalibManager]
            An output calibration manager object. It transforms outputs of the model manager into some
            calibrated version of them.
        """
        super().__init__(
            model_manager, prob_output_layer, output_calib_manager=output_calib_manager
        )

    def _batched_mean(
        self,
        params: Params,
        inputs: Array,
        mutable: Optional[Mutable] = None,
        calib_params: Optional[CalibParams] = None,
        calib_mutable: Optional[CalibMutable] = None,
        **kwargs,
    ) -> jnp.ndarray:
        outputs = super()._get_batched_calibrated_outputs(
            params, inputs, mutable, calib_params, calib_mutable, **kwargs
        )
        return outputs[:, : outputs.shape[1] // 2]

    def _batched_mode(
        self,
        params: Params,
        inputs: Array,
        mutable: Optional[Mutable] = None,
        calib_params: Optional[CalibParams] = None,
        calib_mutable: Optional[CalibMutable] = None,
        **kwargs,
    ) -> jnp.ndarray:
        return self._batched_mean(
            params,
            inputs,
            mutable,
            calib_params=calib_params,
            calib_mutable=calib_mutable,
            **kwargs,
        )

    def _batched_variance(
        self,
        params: Params,
        inputs: Array,
        mutable: Optional[Mutable] = None,
        calib_params: Optional[CalibParams] = None,
        calib_mutable: Optional[CalibMutable] = None,
        **kwargs,
    ) -> jnp.ndarray:
        outputs = super()._get_batched_calibrated_outputs(
            params, inputs, mutable, calib_params, calib_mutable, **kwargs
        )
        return jnp.exp(outputs[:, outputs.shape[1] // 2 :])

    def entropy(
        self,
        params: Params,
        inputs_loader: InputsLoader,
        mutable: Optional[Mutable] = None,
        calib_params: Optional[CalibParams] = None,
        calib_mutable: Optional[CalibMutable] = None,
        n_target_samples: Optional[int] = 30,
        rng: Optional[PRNGKeyArray] = None,
        distribute: bool = True,
        **kwargs,
    ) -> jnp.ndarray:
        samples, aux = self.sample(
            n_target_samples,
            params,
            inputs_loader,
            mutable,
            calib_params=calib_params,
            calib_mutable=calib_mutable,
            return_aux=["outputs"],
            rng=rng,
            distribute=distribute,
        )
        outputs = aux["outputs"]

        @vmap
        def _log_lik_fun(sample: jnp.ndarray):
            return self.prob_output_layer.log_prob(outputs, sample, **kwargs)

        return -jnp.mean(_log_lik_fun(samples), 0)

    def quantile(
        self,
        q: Union[float, jnp.ndarray, np.ndarray],
        params: Optional[Params] = None,
        inputs_loader: Optional[InputsLoader] = None,
        mutable: Optional[Mutable] = None,
        calib_params: Optional[CalibParams] = None,
        calib_mutable: Optional[CalibMutable] = None,
        n_target_samples: Optional[int] = 30,
        target_samples: Optional[jnp.ndarray] = None,
        rng: Optional[PRNGKeyArray] = None,
        distribute: bool = True,
        **kwargs,
    ) -> Union[float, jnp.ndarray]:
        """
        Estimate the `q`-th quantiles of the likelihood function.

        Parameters
        ----------
        q: Union[float, jnp.ndarray, np.ndarray]
            Quantile or sequence of quantiles to compute, which must be between 0 and 1 inclusive.
        params : Params
            The random parameters of the probabilistic model.
        inputs_loader : InputsLoader
            A loader of input data points.
        mutable : Optional[Mutable]
            The mutable objects used to evaluate the models.
        calib_params : Optional[CalibParams]
            The calibration parameters of the probabilistic model.
        calib_mutable : Optional[CalibMutable]
            The calibration mutable objects used to evaluate the calibrators.
        n_target_samples : int
            Number of target samples to sample for each input data point.
        target_samples: Optional[jnp.ndarray] = None
            Samples of the target variable for each input, used to estimate the quantiles.
        rng: Optional[PRNGKeyArray]
            A random number generator. If not passed, this will be taken from the attributes of this class.
        distribute: bool
            Whether to distribute computation over multiple devices, if available.

        Returns
        -------
        jnp.ndarray
            Quantile estimate for each quantile and each input. If multiple quantiles `q` are given, the result's
            first axis is over different quantiles.
        """
        if target_samples is None:
            if params is None or inputs_loader is None:
                raise ValueError(
                    "if `samples` is not passed, then `params` and `inputs_loader` must be passed."
                )
            target_samples = self.sample(
                n_target_samples,
                params,
                inputs_loader,
                mutable,
                calib_params=calib_params,
                calib_mutable=calib_mutable,
                rng=rng,
                distribute=distribute,
                **kwargs,
            )
        return jnp.quantile(target_samples, q, axis=0)
