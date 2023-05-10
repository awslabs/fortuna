from typing import Optional

import jax
import jax.numpy as jnp
from jax import vmap

from fortuna.data.loader import InputsLoader
from fortuna.likelihood.base import Likelihood
from fortuna.model.model_manager.classification import ClassificationModelManager
from fortuna.output_calibrator.output_calib_manager.base import OutputCalibManager
from fortuna.prob_output_layer.classification import ClassificationProbOutputLayer
from fortuna.typing import Array, CalibMutable, CalibParams, Mutable, Params


class ClassificationLikelihood(Likelihood):
    def __init__(
        self,
        model_manager: ClassificationModelManager,
        prob_output_layer: ClassificationProbOutputLayer,
        output_calib_manager: Optional[OutputCalibManager] = None,
    ):
        """
        A classification likelihood function class. In this class, the likelihood function is additionally assumed to
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
            A probabilistic output layer object. This object characterizes the probability distribution of the
            target variable given the calibrated outputs.
        output_calib_manager : Optional[OutputCalibManager]
            An output calibration manager object. It transforms outputs of the model manager into some
            calibrated version of them.
        """
        super().__init__(
            model_manager, prob_output_layer, output_calib_manager=output_calib_manager
        )

    def mean(
        self,
        params: Params,
        inputs_loader: InputsLoader,
        mutable: Optional[Mutable] = None,
        calib_params: Optional[CalibParams] = None,
        calib_mutable: Optional[CalibMutable] = None,
        distribute: bool = True,
        **kwargs
    ) -> jnp.ndarray:
        r"""
        Estimate the likelihood mean of the one-hot encoded target variable, that is

        .. math::
            \mathbb{E}_{\tilde{Y}|w, x}[\tilde{Y}],

        where:
         - :math:`x` is an observed input variable;
         - :math:`\tilde{Y}` is a one-hot encoded random target variable;
         - :math:`w` denotes the observed model parameters.

        Parameters
        ----------
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
        distribute: bool
            Whether to distribute computation over multiple devices, if available.

        Returns
        -------
        jnp.ndarray
            An estimate of the likelihood mean for each input.
        """
        return super().mean(
            params,
            inputs_loader,
            mutable,
            calib_params,
            calib_mutable,
            distribute,
            **kwargs
        )

    def _batched_mean(
        self,
        params: Params,
        inputs: Array,
        mutable: Optional[Mutable] = None,
        calib_params: Optional[CalibParams] = None,
        calib_mutable: Optional[CalibMutable] = None,
        **kwargs
    ) -> jnp.ndarray:
        outputs = self._get_batched_calibrated_outputs(
            params, inputs, mutable, calib_params, calib_mutable, **kwargs
        )
        return jax.nn.softmax(outputs, -1)

    def _batched_mode(
        self,
        params: Params,
        inputs: Array,
        mutable: Optional[Mutable] = None,
        calib_params: Optional[CalibParams] = None,
        calib_mutable: Optional[CalibMutable] = None,
        **kwargs
    ) -> jnp.ndarray:
        outputs = self._get_batched_calibrated_outputs(
            params, inputs, mutable, calib_params, calib_mutable, **kwargs
        )
        return jnp.argmax(outputs, -1)

    def variance(
        self,
        params: Params,
        inputs_loader: InputsLoader,
        mutable: Optional[Mutable] = None,
        calib_params: Optional[CalibParams] = None,
        calib_mutable: Optional[CalibMutable] = None,
        distribute: bool = True,
        **kwargs
    ) -> jnp.ndarray:
        r"""
        Estimate the likelihood variance of the one-hot encoded target variable, that is

        .. math::
            \text{Var}_{\tilde{Y}|w,x}[\tilde{Y}],

        where:
         - :math:`x` is an observed input variable;
         - :math:`\tilde{Y}` is a one-hot encoded random target variable;
         - :math:`w` denotes the observed model parameters.

        Parameters
        ----------
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
        distribute: bool
            Whether to distribute computation over multiple devices, if available.

        Returns
        -------
        jnp.ndarray
            An estimate of the likelihood variance for each input.
        """
        return super().variance(
            params,
            inputs_loader,
            mutable,
            calib_params,
            calib_mutable,
            distribute,
            **kwargs
        )

    def _batched_variance(
        self,
        params: Params,
        inputs: Array,
        mutable: Optional[Mutable] = None,
        calib_params: Optional[CalibParams] = None,
        calib_mutable: Optional[CalibMutable] = None,
        distribute: bool = True,
        **kwargs
    ) -> jnp.ndarray:
        means = self._batched_mean(
            params=params,
            inputs=inputs,
            mutable=mutable,
            calib_params=calib_params,
            calib_mutable=calib_mutable,
            **kwargs
        )
        return means * (1 - means)

    def entropy(
        self,
        params: Params,
        inputs_loader: InputsLoader,
        mutable: Optional[Mutable] = None,
        calib_params: Optional[CalibParams] = None,
        calib_mutable: Optional[CalibMutable] = None,
        distribute: bool = True,
        **kwargs
    ) -> jnp.ndarray:
        outputs = super().get_calibrated_outputs(
            params, inputs_loader, mutable, calib_params, calib_mutable, distribute
        )
        n_classes = outputs.shape[-1]

        @vmap
        def _entropy_term(i: int):
            targets = i * jnp.ones(outputs.shape[0])
            log_liks = self.prob_output_layer.log_prob(outputs, targets, **kwargs)
            return jnp.exp(log_liks) * log_liks

        return -jnp.sum(_entropy_term(jnp.arange(n_classes)), 0)
