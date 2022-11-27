import abc
from typing import Any, List, Optional, Tuple, Union, Callable

import jax
import jax.numpy as jnp
from jax import jit, pmap
from jax._src.prng import PRNGKeyArray

from fortuna.data.loader import DataLoader, InputsLoader, DeviceDimensionAugmentedDataLoader, \
    DeviceDimensionAugmentedInputsLoader
from fortuna.model.model_manager.base import ModelManager
from fortuna.output_calibrator.output_calib_manager.base import \
    OutputCalibManager
from fortuna.prob_output_layer.base import ProbOutputLayer
from fortuna.typing import (Array, Batch, CalibMutable, CalibParams, Mutable,
                            Params)
from fortuna.utils.random import WithRNG


class Likelihood(WithRNG):
    def __init__(
        self,
        model_manager: ModelManager,
        prob_output_layer: ProbOutputLayer,
        output_calib_manager: OutputCalibManager,
    ):
        """
        A likelihood function abstract class. In this class, the likelihood function is additionally assumed to be a
        probability density function, i.e. positive and integrating to 1. The likelihood is formed by three objects
        applied in sequence: the model manager, the output calibrator and the probabilistic output layer. The output
        maker maps parameters and inputs to outputs. The output calibration takes outputs and returns some calibrated
        version of them. The probabilistic output layer describes the probability distribution of the calibrated
        outputs.

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
        self.model_manager = model_manager
        self.prob_output_layer = prob_output_layer
        self.output_calib_manager = output_calib_manager

    def log_prob(
        self,
        params: Params,
        data_loader: DataLoader,
        mutable: Optional[Mutable] = None,
        calib_params: Optional[CalibParams] = None,
        calib_mutable: Optional[CalibMutable] = None,
        distribute: bool = True,
        **kwargs
    ) -> jnp.ndarray:
        """
        Evaluate the log-likelihood function.

        Parameters
        ----------
        params : Params
            The random parameters of the probabilistic model.
        data_loader : DataLoader
            A data loader.
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
            The evaluation of the log-likelihood function. 
        """
        return self._loop_fun_through_data_loader(self._batched_log_prob, params, data_loader, mutable, calib_params,
                                                  calib_mutable, distribute, **kwargs)

    def _batched_log_prob(
        self,
        params: Params,
        batch: Batch,
        mutable: Optional[Mutable] = None,
        calib_params: Optional[CalibParams] = None,
        calib_mutable: Optional[CalibMutable] = None,
        **kwargs
    ) -> jnp.ndarray:
        outputs = self._get_batched_calibrated_outputs(params, batch[0], mutable, calib_params, calib_mutable, **kwargs)
        return self.prob_output_layer.log_prob(outputs, batch[1], **kwargs)

    def _batched_log_joint_prob(
        self,
        params: Params,
        batch: Batch,
        n_data: int,
        mutable: Optional[Mutable] = None,
        calib_params: Optional[CalibParams] = None,
        calib_mutable: Optional[CalibMutable] = None,
        return_aux: Optional[List[str]] = None,
        train: bool = False,
        outputs: Optional[jnp.ndarray] = None,
        rng: Optional[PRNGKeyArray] = None,
        **kwargs
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Any]]:
        """
        Evaluate the batched log-likelihood function.

        Parameters
        ----------
        params : Params
            The random parameters of the probabilistic model.
        batch : Batch
            A batch of data points.
        n_data : int
            The total number of data points over which the likelihood is joint. This is used to rescale the batched
            log-likelihood function to better approximate the full likelihood.
        mutable : Optional[Mutable]
            The mutable objects used to evaluate the models.
        calib_params : Optional[CalibParams]
            The calibration parameters of the probabilistic model.
        calib_mutable : Optional[CalibMutable]
            The calibration mutable objects used to evaluate the calibrators.
        return_aux : Optional[List[str]]
            The auxiliary objects to return. We support 'outputs', 'mutable' and 'calib_mutable'. If this argument is
            not given, no auxiliary object is returned.
        train : bool
            Whether the method is called during training.
        outputs : Optional[jnp.ndarray]
            Pre-computed batch of outputs.
        rng: Optional[PRNGKeyArray]
            A random number generator. If not passed, this will be taken from the attributes of this class.

        Returns
        -------
        Union[jnp.ndarray, Tuple[jnp.ndarray, Any]]
            The evaluation of the batched log-likelihood function. If `return_aux` is given, the corresponding
            auxiliary objects are also returned.
        """
        if return_aux is None:
            return_aux = []
        supported_aux = ["outputs", "mutable", "calib_mutable"]
        unsupported_aux = [s for s in return_aux if s not in supported_aux]
        if sum(unsupported_aux) > 0:
            raise AttributeError(
                """The auxiliary objects {} is unknown. Please make sure that all elements of `return_aux` 
                            belong to the following list: {}""".format(
                    unsupported_aux, supported_aux
                )
            )
        if train and outputs is not None:
            raise ValueError(
                """When `outputs` is available, `train` must be set to `False`."""
            )
        if "mutable" in return_aux and outputs is not None:
            raise ValueError(
                """When `outputs` is available, `return_aux` cannot contain 'mutable'`."""
            )
        if not train and "mutable" in return_aux:
            raise ValueError(
                "Returning an auxiliary mutable is supported only during training. Please either set `train` to "
                "`True`, or remove 'mutable' from `return_aux`."
            )
        if "mutable" in return_aux and mutable is None:
            raise ValueError(
                "In order to be able to return an auxiliary mutable, an initial mutable must be passed as `mutable`. "
                "Please either remove 'mutable' from `return_aux`, or pass an initial mutable as `mutable`."
            )
        if "mutable" not in return_aux and mutable is not None and train is True:
            raise ValueError(
                """You need to add `mutable` to `return_aux`. When you provide a (not null) `mutable` variable during 
                training, that variable will be updated during the forward pass."""
            )

        inputs, targets = batch
        if outputs is None:
            outs = self.model_manager.apply(
                params, inputs, train=train, mutable=mutable, rng=rng,
            )
            if "mutable" in return_aux:
                outputs, aux = outs
                mutable = aux["mutable"]
            else:
                outputs = outs

        aux = dict()
        outs = self.output_calib_manager.apply(
            params=calib_params["output_calibrator"]
            if calib_params is not None
            else None,
            mutable=calib_mutable["output_calibrator"]
            if calib_mutable is not None
            else None,
            outputs=outputs,
            calib="calib_mutable" in return_aux,
        )
        if (
            calib_mutable is not None
            and calib_mutable["output_calibrator"] is not None
            and "calib_mutable" in return_aux
        ):
            outputs, aux["calib_mutable"] = outs
            aux["calib_mutable"] = dict(output_calibrator=aux["calib_mutable"])
        else:
            outputs = outs
            if "calib_mutable" in return_aux:
                aux["calib_mutable"] = dict(output_calibrator=None)

        log_joint_prob = jnp.sum(self.prob_output_layer.log_prob(outputs, targets, **kwargs))
        batch_weight = n_data / targets.shape[0]
        log_joint_prob *= batch_weight

        if len(return_aux) == 0:
            return log_joint_prob
        else:
            if "outputs" in return_aux:
                aux["outputs"] = outputs
            if "mutable" in return_aux:
                aux["mutable"] = mutable
            return log_joint_prob, aux

    def sample(
        self,
        n_target_samples: int,
        params: Params,
        inputs_loader: InputsLoader,
        mutable: Optional[Mutable] = None,
        calib_params: Optional[CalibParams] = None,
        calib_mutable: Optional[CalibMutable] = None,
        return_aux: Optional[List[str]] = None,
        rng: Optional[PRNGKeyArray] = None,
        distribute: bool = True,
        **kwargs
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, dict]]:
        """
        Sample target variables from the likelihood function for each input variable.

        Parameters
        ----------
        n_target_samples : int
            The number of samples to draw from the likelihood for each input data point.
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
        return_aux : Optional[List[str]]
            The auxiliary objects to return. We support 'outputs'. If this argument is not given, no auxiliary object
            is returned.
        rng: Optional[PRNGKeyArray]
            A random number generator. If not passed, this will be taken from the attributes of this class.
        distribute: bool
            Whether to distribute computation over multiple devices, if available.

        Returns
        -------
        Union[jnp.ndarray, Tuple[jnp.ndarray, dict]]
            The samples of the target variable for each input. If `return_aux` is given, the corresponding auxiliary
            objects are also returned.
        """
        if return_aux is None:
            return_aux = []
        supported_aux = ["outputs"]
        unsupported_aux = [s for s in return_aux if s not in supported_aux]
        if sum(unsupported_aux) > 0:
            raise Exception(
                """The auxiliary objects {} are unknown. Please make sure that all elements of `return_aux` 
                            belong to the following list: {}""".format(
                    unsupported_aux, supported_aux
                )
            )

        outputs = self.get_calibrated_outputs(params, inputs_loader, mutable, calib_params, calib_mutable, distribute, **kwargs)

        samples = self.prob_output_layer.sample(
            n_target_samples, outputs, rng=rng, **kwargs
        )
        if len(return_aux) > 0:
            return samples, dict(outputs=outputs)
        return samples

    def _batched_sample(
        self,
        n_target_samples: int,
        params: Params,
        inputs: Array,
        mutable: Optional[Mutable] = None,
        calib_params: Optional[CalibParams] = None,
        calib_mutable: Optional[CalibMutable] = None,
        return_aux: Optional[List[str]] = None,
        rng: Optional[PRNGKeyArray] = None,
        **kwargs
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, dict]]:
        if return_aux is None:
            return_aux = []
        supported_aux = ["outputs"]
        unsupported_aux = [s for s in return_aux if s not in supported_aux]
        if sum(unsupported_aux) > 0:
            raise Exception(
                """The auxiliary objects {} are unknown. Please make sure that all elements of `return_aux` 
                            belong to the following list: {}""".format(
                    unsupported_aux, supported_aux
                )
            )

        outputs = self._get_batched_calibrated_outputs(params, inputs, mutable, calib_params, calib_mutable, **kwargs)

        samples = self.prob_output_layer.sample(
            n_target_samples, outputs, rng=rng, **kwargs
        )
        if len(return_aux) > 0:
            return samples, dict(outputs=outputs)
        return samples

    def get_calibrated_outputs(
        self,
        params: Params,
        inputs_loader: InputsLoader,
        mutable: Optional[Mutable] = None,
        calib_params: Optional[CalibParams] = None,
        calib_mutable: Optional[CalibMutable] = None,
        distribute: bool = True,
        **kwargs
    ) -> jnp.ndarray:
        """
        Compute the outputs and their calibrated version.

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
            The calibrated outputs.
        """
        outputs = self.get_outputs(params, inputs_loader, mutable, distribute, **kwargs)

        if self.output_calib_manager is not None:
            outputs = self.output_calib_manager.apply(
                params=calib_params["output_calibrator"]
                if calib_params is not None
                else None,
                mutable=calib_mutable["output_calibrator"]
                if calib_mutable is not None
                else None,
                outputs=outputs,
                **kwargs
            )
        return outputs

    def _get_batched_calibrated_outputs(
        self,
        params: Params,
        inputs: Array,
        mutable: Optional[Mutable] = None,
        calib_params: Optional[CalibParams] = None,
        calib_mutable: Optional[CalibMutable] = None,
        **kwargs
    ) -> jnp.ndarray:
        outputs = self.model_manager.apply(params, inputs, mutable, **kwargs)

        if self.output_calib_manager is not None:
            outputs = self.output_calib_manager.apply(
                params=calib_params["output_calibrator"]
                if calib_params is not None
                else None,
                mutable=calib_mutable["output_calibrator"]
                if calib_mutable is not None
                else None,
                outputs=outputs,
                **kwargs
            )
        return outputs

    def get_outputs(
        self,
        params: Params,
        inputs_loader: InputsLoader,
        mutable: Optional[Mutable] = None,
        distribute: bool = True,
        **kwargs
    ) -> jnp.ndarray:
        """
        Compute the outputs and their calibrated version.

        Parameters
        ----------
        params : Params
            The random parameters of the probabilistic model.
        inputs_loader : InputsLoader
            A loader of input data points.
        mutable : Optional[Mutable]
            The mutable objects used to evaluate the models.
        distribute: bool
            Whether to distribute computation over multiple devices, if available.

        Returns
        -------
        jnp.ndarray
            The calibrated outputs.
        """
        if distribute and jax.local_device_count() <= 1:
            distribute = False

        if distribute:
            inputs_loader = DeviceDimensionAugmentedInputsLoader(inputs_loader)

        @jit
        def fun(_inputs):
            return self.model_manager.apply(params, _inputs, mutable, **kwargs)

        outputs = []
        for inputs in inputs_loader:
            outputs.append(self._unshard_array(pmap(fun)(inputs)) if distribute else fun(inputs))
        return jnp.concatenate(outputs, 0)

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
        Estimate the likelihood mean of the target variable, that is

        .. math::
            \mathbb{E}_{Y|w, x}[Y],

        where:
         - :math:`x` is an observed input variable;
         - :math:`Y` is a random target variable;
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
        return self._loop_fun_through_inputs_loader(self._batched_mean, params, inputs_loader, mutable, calib_params, calib_mutable, distribute, **kwargs)

    @abc.abstractmethod
    def _batched_mean(
        self,
        params: Params,
        inputs: Array,
        mutable: Optional[Mutable] = None,
        calib_params: Optional[CalibParams] = None,
        calib_mutable: Optional[CalibMutable] = None,
        **kwargs
    ) -> jnp.ndarray:
        pass

    def mode(
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
        Estimate the likelihood mode of the target variable, that is

        .. math::
            \text{argmax}_y\ p(y|w, x),

        where:
         - :math:`x` is an observed input variable;
         - :math:`w` denotes the observed model parameters;
         - :math:`y` is the target variable to optimize upon.

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
            An estimate of the likelihood mode for each input.
        """
        return self._loop_fun_through_inputs_loader(self._batched_mode, params, inputs_loader, mutable, calib_params, calib_mutable, distribute, **kwargs)

    @abc.abstractmethod
    def _batched_mode(
        self,
        params: Params,
        inputs: Array,
        mutable: Optional[Mutable] = None,
        calib_params: Optional[CalibParams] = None,
        calib_mutable: Optional[CalibMutable] = None,
        **kwargs
    ) -> jnp.ndarray:
        pass

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
        Estimate the likelihood variance of the target variable, that is

        .. math::
            \text{Var}_{Y|w,x}[Y],

        where:
         - :math:`x` is an observed input variable;
         - :math:`Y` is a random target variable;
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
        return self._loop_fun_through_inputs_loader(self._batched_variance, params, inputs_loader, mutable, calib_params, calib_mutable, distribute, **kwargs)

    @abc.abstractmethod
    def _batched_variance(
        self,
        params: Params,
        inputs: Array,
        mutable: Optional[Mutable] = None,
        calib_params: Optional[CalibParams] = None,
        calib_mutable: Optional[CalibMutable] = None,
        **kwargs
    ) -> jnp.ndarray:
        pass

    def std(
        self,
        params: Params,
        inputs_loader: InputsLoader,
        mutable: Optional[Mutable] = None,
        calib_params: Optional[CalibParams] = None,
        calib_mutable: Optional[CalibMutable] = None,
        variance: Optional[jnp.ndarray] = None,
        distribute: bool = True,
        **kwargs
    ) -> jnp.ndarray:
        r"""
        Estimate the likelihood standard deviation of the target variable, that is

        .. math::
            \sqrt{\text{Var}_{Y|w,x}[Y]},

        where:
         - :math:`x` is an observed input variable;
         - :math:`Y` is a random target variable;
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
        variance : Optional[jnp.ndarray]
            An estimate of the likelihood variance for each input.
        distribute: bool
            Whether to distribute computation over multiple devices, if available.

        Returns
        -------
        jnp.ndarray
            An estimate of the likelihood standard deviation for each input.
        """
        if variance is None:
            variance = self.variance(
                params,
                inputs_loader,
                mutable,
                calib_params=calib_params,
                calib_mutable=calib_mutable,
                distribute=distribute,
                **kwargs
            )
        return jnp.sqrt(variance)

    @abc.abstractmethod
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
        r"""
        Estimate the likelihood entropy, that is

        .. math::
            -\mathbb{E}_{Y|w,x}[\log p(Y|w,x)]

        where:
         - :math:`x` is an observed input variable;
         - :math:`Y` is a random target variable;
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
            An estimate of the likelihood entropy for each input.
        """
        pass

    @staticmethod
    def _unshard_array(arr: Array) -> Array:
        return arr.reshape((arr.shape[0] * arr.shape[1],) + arr.shape[2:])

    def _loop_fun_through_inputs_loader(
            self,
            fun: Callable,
            params: Params,
            inputs_loader: InputsLoader,
            mutable: Optional[Mutable] = None,
            calib_params: Optional[CalibParams] = None,
            calib_mutable: Optional[CalibMutable] = None,
            distribute: bool = True,
            **kwargs
    ) -> Array:
        if distribute and jax.local_device_count() <= 1:
            distribute = False

        def fun2(_inputs):
            return fun(params, _inputs, mutable, calib_params, calib_mutable, **kwargs)

        if distribute:
            inputs_loader = DeviceDimensionAugmentedInputsLoader(inputs_loader)
            fun2 = pmap(fun2)
            return jnp.concatenate([self._unshard_array(fun2(inputs)) for inputs in inputs_loader], 0)
        fun2 = jit(fun2)
        return jnp.concatenate([fun2(inputs) for inputs in inputs_loader], 0)

    def _loop_fun_through_data_loader(
            self,
            fun: Callable,
            params: Params,
            data_loader: DataLoader,
            mutable: Optional[Mutable] = None,
            calib_params: Optional[CalibParams] = None,
            calib_mutable: Optional[CalibMutable] = None,
            distribute: bool = True,
            **kwargs
    ) -> Array:
        if distribute and jax.local_device_count() <= 1:
            distribute = False

        def fun2(_batch):
            return fun(params, _batch, mutable, calib_params, calib_mutable, **kwargs)

        if distribute:
            data_loader = DeviceDimensionAugmentedDataLoader(data_loader)
            fun2 = pmap(fun2)
            return jnp.concatenate([self._unshard_array(fun2(batch)) for batch in data_loader], 0)
        fun2 = jit(fun2)
        return jnp.concatenate([fun2(batch) for batch in data_loader], 0)
