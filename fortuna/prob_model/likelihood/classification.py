from typing import Optional

import jax
import jax.numpy as jnp
from jax import vmap

from fortuna.data.loader import InputsLoader
from fortuna.model.model_manager.classification import \
    ClassificationModelManager
from fortuna.output_calibrator.output_calib_manager.base import \
    OutputCalibManager
from fortuna.prob_model.likelihood.base import Likelihood
from fortuna.prob_output_layer.classification import \
    ClassificationProbOutputLayer
from fortuna.typing import CalibMutable, CalibParams, Mutable, Params


class ClassificationLikelihood(Likelihood):
    def __init__(
        self,
        model_manager: ClassificationModelManager,
        prob_output_layer: ClassificationProbOutputLayer,
        output_calib_manager: OutputCalibManager,
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
        output_calib_manager : OutputCalibManager
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

        Returns
        -------
        jnp.ndarray
            An estimate of the likelihood mean for each input.
        """
        outputs = []
        for batch_inputs in inputs_loader:
            outputs.append(
                self.model_manager.apply(params, batch_inputs, mutable, **kwargs)
            )
        outputs = jnp.concatenate(outputs, 0)

        outputs = self.output_calib_manager.apply(
            params=calib_params["output_calibrator"]
            if calib_params is not None
            else None,
            mutable=calib_mutable["output_calibrator"]
            if calib_mutable is not None
            else None,
            outputs=outputs,
        )
        return jax.nn.softmax(outputs, -1)

    def mode(
        self,
        params: Params,
        inputs_loader: InputsLoader,
        mutable: Optional[Mutable] = None,
        calib_params: Optional[CalibParams] = None,
        calib_mutable: Optional[CalibMutable] = None,
        **kwargs
    ) -> jnp.ndarray:
        outputs = []
        for batch_inputs in inputs_loader:
            outputs.append(
                self.model_manager.apply(params, batch_inputs, mutable, **kwargs)
            )
        outputs = jnp.concatenate(outputs, 0)

        outputs = self.output_calib_manager.apply(
            params=calib_params["output_calibrator"]
            if calib_params is not None
            else None,
            mutable=calib_mutable["output_calibrator"]
            if calib_mutable is not None
            else None,
            outputs=outputs,
        )
        return jnp.argmax(outputs, -1)

    def variance(
        self,
        params: Params,
        inputs_loader: InputsLoader,
        mutable: Optional[Mutable] = None,
        calib_params: Optional[CalibParams] = None,
        calib_mutable: Optional[CalibMutable] = None,
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

        Returns
        -------
        jnp.ndarray
            An estimate of the likelihood variance for each input.
        """
        means = self.mean(
            params=params,
            inputs_loader=inputs_loader,
            mutable=mutable,
            calib_params=calib_params,
            calib_mutable=calib_mutable,
        )
        return means * (1 - means)

    def entropy(
        self,
        params: Params,
        inputs_loader: InputsLoader,
        mutable: Optional[Mutable] = None,
        calib_params: Optional[CalibParams] = None,
        calib_mutable: Optional[CalibMutable] = None,
        **kwargs
    ) -> jnp.ndarray:
        outputs = []
        for batch_inputs in inputs_loader:
            outputs.append(
                self.model_manager.apply(params, batch_inputs, mutable, **kwargs)
            )
        outputs = jnp.concatenate(outputs, 0)
        n_classes = outputs.shape[-1]

        outputs = self.output_calib_manager.apply(
            params=calib_params["output_calibrator"]
            if calib_params is not None
            else None,
            mutable=calib_mutable["output_calibrator"]
            if calib_mutable is not None
            else None,
            outputs=outputs,
        )

        @vmap
        def _entropy_term(i: int):
            targets = i * jnp.ones(outputs.shape[0])
            log_liks = self.prob_output_layer.log_prob(outputs, targets, **kwargs)
            return jnp.exp(log_liks) * log_liks

        return -jnp.sum(_entropy_term(jnp.arange(n_classes)), 0)
