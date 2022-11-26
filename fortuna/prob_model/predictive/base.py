import abc
from typing import List, Optional, Tuple, Union

import jax.numpy as jnp
import jax.scipy as jsp
from jax import lax, random
from jax._src.prng import PRNGKeyArray
from jax.tree_util import tree_map

from fortuna.data.loader import DataLoader, InputsLoader, TargetsLoader
from fortuna.prob_model.posterior.base import Posterior
from fortuna.typing import Batch, CalibMutable, CalibParams
from fortuna.utils.random import WithRNG


class Predictive(WithRNG):
    def __init__(self, posterior: Posterior):
        """
        Predictive distribution abstract class.

        Parameters
        ----------
        posterior : Posterior
             A posterior distribution object.
        """
        self.likelihood = posterior.joint.likelihood
        self.posterior = posterior

    def log_prob(
        self,
        data_loader: DataLoader,
        n_posterior_samples: int = 30,
        return_aux: Optional[List[str]] = None,
        ensemble_outputs: Optional[jnp.ndarray] = None,
        calib_params: Optional[CalibParams] = None,
        calib_mutable: Optional[CalibMutable] = None,
        rng: Optional[PRNGKeyArray] = None,
        **kwargs
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, dict]]:
        r"""
        Estimate the predictive log-probability density function (a.k.a. log-pdf), that is

        .. math::
            \log p(y|x, \mathcal{D}),

        where:
         - :math:`x` is an observed input variable;
         - :math:`y` is an observed target variable;
         - :math:`\mathcal{D}` is the observed training data set.

        Parameters
        ----------
        data_loader : DataLoader
            A data loader.
        n_posterior_samples : int
            Number of posterior samples to draw in order to approximate the predictive log-pdf.
        return_aux : Optional[List[str]]
            Which auxiliary outputs to return. We currently support 'outputs' and 'calib_mutable'.
        ensemble_outputs : Optional[jnp.ndarray]
            Pre-computed ensemble of outputs. If this argument is passed, these outputs are used instead of the outputs
            that would be produced using the posterior distribution state.
        calib_params : Optional[CalibParams]
            The calibration parameters of the probabilistic model.
        calib_mutable : Optional[CalibMutable]
            The calibration mutable objects used to evaluate the calibrators.
        rng : Optional[PRNGKeyArray]
            A random number generator. If not passed, this will be taken from the attributes of this class.

        Returns
        -------
        jnp.ndarray
            An estimate of the predictive log-pdf for each data point.
        """
        if return_aux is None:
            return_aux = []
        supported_aux = ["outputs", "calib_mutable"]
        unsupported_aux = [s for s in return_aux if s not in supported_aux]
        if sum(unsupported_aux) > 0:
            raise Exception(
                """The auxiliary objects {} are unknown. Please make sure that all elements of `return_aux` 
                            belong to the following list: {}""".format(
                    unsupported_aux, supported_aux
                )
            )

        inputs_loader = data_loader.to_inputs_loader()

        if ensemble_outputs is None:
            if rng is None:
                rng = self.rng.get()
            keys = random.split(rng, n_posterior_samples)

            def _lik_log_prob(key):
                sample = self.posterior.sample(inputs_loader=inputs_loader, rng=key)
                return self.likelihood.log_prob(
                    sample.params,
                    data_loader,
                    mutable=sample.mutable,
                    return_aux=return_aux,
                    calib_params=calib_params
                    if calib_params is not None
                    else sample.calib_params,
                    calib_mutable=calib_mutable
                    if calib_mutable is not None
                    else sample.calib_mutable,
                    **kwargs
                )

            outs = lax.map(_lik_log_prob, keys)
        else:
            if calib_params is None:
                d = self.posterior.state.extract_calib_keys()
                calib_params = d["calib_params"]
                calib_mutable = d["calib_mutable"]

            def _lik_log_prob(outputs):
                return self.likelihood.log_prob(
                    None,
                    data_loader,
                    mutable=None,
                    outputs=outputs,
                    return_aux=return_aux,
                    calib_params=calib_params,
                    calib_mutable=calib_mutable,
                    **kwargs
                )

            outs = lax.map(_lik_log_prob, ensemble_outputs)

        if len(return_aux):
            ensemble_log_lik, aux = outs
            aux = tree_map(lambda v: jnp.mean(v, 0), aux)
        else:
            ensemble_log_lik = outs
        log_pred = jsp.special.logsumexp(ensemble_log_lik) - jnp.log(
            len(ensemble_log_lik)
        )
        if len(return_aux):
            return log_pred, aux
        return log_pred

    def _batched_log_prob(
        self,
        batch: Batch,
        n_data: int,
        n_posterior_samples: int = 30,
        return_aux: Optional[List[str]] = None,
        ensemble_outputs: Optional[jnp.ndarray] = None,
        calib_params: Optional[CalibParams] = None,
        calib_mutable: Optional[CalibMutable] = None,
        rng: Optional[PRNGKeyArray] = None,
        **kwargs
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, dict]]:
        if return_aux is None:
            return_aux = []
        supported_aux = ["outputs", "calib_mutable"]
        unsupported_aux = [s for s in return_aux if s not in supported_aux]
        if sum(unsupported_aux) > 0:
            raise Exception(
                """The auxiliary objects {} are unknown. Please make sure that all elements of `return_aux` 
                            belong to the following list: {}""".format(
                    unsupported_aux, supported_aux
                )
            )

        if ensemble_outputs is None:
            if rng is None:
                rng = self.rng.get()
            keys = random.split(rng, n_posterior_samples)

            def _lik_log_prob(key):
                sample = self.posterior.sample(inputs=batch[0], rng=key)
                return self.likelihood._batched_log_prob(
                    sample.params,
                    batch,
                    n_data,
                    mutable=sample.mutable,
                    return_aux=return_aux,
                    calib_params=calib_params
                    if calib_params is not None
                    else sample.calib_params,
                    calib_mutable=calib_mutable
                    if calib_mutable is not None
                    else sample.calib_mutable,
                    **kwargs
                )

            outs = lax.map(_lik_log_prob, keys)
        else:
            if calib_params is None:
                d = self.posterior.state.extract_calib_keys()
                calib_params = d["calib_params"]
                calib_mutable = d["calib_mutable"]

            def _lik_log_prob(outputs):
                return self.likelihood._batched_log_prob(
                    None,
                    batch,
                    n_data,
                    mutable=None,
                    outputs=outputs,
                    return_aux=return_aux,
                    calib_params=calib_params,
                    calib_mutable=calib_mutable,
                    **kwargs
                )

            outs = lax.map(_lik_log_prob, ensemble_outputs)

        if len(return_aux):
            ensemble_log_lik, aux = outs
            aux = tree_map(lambda v: jnp.mean(v, 0), aux)
        else:
            ensemble_log_lik = outs
        log_pred = jsp.special.logsumexp(ensemble_log_lik) - jnp.log(
            len(ensemble_log_lik)
        )
        if len(return_aux):
            return log_pred, aux
        return log_pred

    def sample(
        self,
        inputs_loader: InputsLoader,
        n_target_samples: int = 1,
        return_aux: Optional[List[str]] = None,
        rng: Optional[PRNGKeyArray] = None,
        **kwargs
    ) -> jnp.ndarray:
        r"""
        Sample from an approximation of the predictive distribution for each input data point, that is

        .. math::
            y^{(i)}\sim p(\cdot|x, \mathcal{D}),

        where:
         - :math:`x` is an observed input variable;
         - :math:`\mathcal{D}` is the observed training data set;
         - :math:`y^{(i)}` is a sample of the target variable for the input :math:`x`.

        Parameters
        ----------
        inputs_loader : InputsLoader
            A loader of input data points.
        n_target_samples : int
            Number of target samples to sample for each input data point.
        return_aux : Optional[List[str]]
            Return auxiliary objects. We currently support 'outputs'.
        rng : Optional[PRNGKeyArray]
            A random number generator. If not passed, this will be taken from the attributes of this class.

        Returns
        -------
        jnp.ndarray
            Samples for each input data point.
        """
        if return_aux is None:
            return_aux = []

        if not rng:
            rng = self.rng.get()
        keys = random.split(rng, n_target_samples)

        def _sample(key):
            key1, key2 = random.split(key, 2)
            _post_sample = self.posterior.sample(inputs_loader=inputs_loader, rng=key1)
            outs = self.likelihood.sample(
                1,
                _post_sample.params,
                inputs_loader,
                mutable=_post_sample.mutable,
                calib_params=_post_sample.calib_params,
                calib_mutable=_post_sample.calib_mutable,
                return_aux=return_aux,
                rng=key2,
                **kwargs
            )
            if len(return_aux) > 0:
                _samples, aux = outs
                return _samples.squeeze(0), aux
            return outs.squeeze(0)

        return lax.map(_sample, keys)

    def _sample_calibrated_outputs(
        self,
        inputs_loader: InputsLoader,
        n_output_samples: int = 1,
        rng: Optional[PRNGKeyArray] = None,
    ) -> jnp.ndarray:
        r"""
        Sample parameters from the posterior distribution state and compute calibrated outputs.

        Parameters
        ----------
        inputs_loader : InputsLoader
            A loader of input data points.
        n_output_samples : int
            Number of output samples to draw for each input.
        rng: Optional[PRNGKeyArray]
            A random number generator. If not passed, this will be taken from the attributes of this class.

        Returns
        -------
        jnp.ndarray
            Samples of calibrated outputs.
        """
        if rng is None:
            rng = self.rng.get()
        keys = random.split(rng, n_output_samples)

        def _sample(key):
            sample = self.posterior.sample(inputs_loader=inputs_loader, rng=key)
            return self.likelihood._get_calibrated_outputs(
                params=sample.params,
                inputs_loader=inputs_loader,
                mutable=sample.mutable,
                calib_params=sample.calib_params,
                calib_mutable=sample.calib_mutable,
            )

        return lax.map(_sample, keys)

    def _sample_outputs(
        self,
        inputs_loader: InputsLoader,
        n_output_samples: int = 1,
        rng: Optional[PRNGKeyArray] = None,
    ) -> jnp.ndarray:
        r"""
        Sample parameters from the posterior distribution state and compute calibrated outputs.

        Parameters
        ----------
        inputs_loader : InputsLoader
            A loader of input data points.
        n_output_samples : int
            Number of output samples to draw for each input.
        rng: Optional[PRNGKeyArray]
            A random number generator. If not passed, this will be taken from the attributes of this class.

        Returns
        -------
        jnp.ndarray
            Samples of calibrated outputs.
        """
        if rng is None:
            rng = self.rng.get()
        keys = random.split(rng, n_output_samples)

        def _sample(key):
            sample = self.posterior.sample(inputs_loader=inputs_loader, rng=key)
            return self.likelihood._get_outputs(
                params=sample.params,
                inputs_loader=inputs_loader,
                mutable=sample.mutable,
            )

        return lax.map(_sample, keys)

    def _sample_outputs_loader(
        self,
        inputs_loader: InputsLoader,
        n_output_samples: int = 1,
        rng: Optional[PRNGKeyArray] = None,
        return_size: bool = False,
    ) -> Union[TargetsLoader, Tuple[TargetsLoader, int]]:
        if rng is None:
            rng = self.rng.get()
        keys = random.split(rng, n_output_samples)

        def _sample(key, _inputs):
            sample = self.posterior.sample(inputs=_inputs, rng=key)
            return self.likelihood._get_batched_outputs(
                params=sample.params, inputs=_inputs, mutable=sample.mutable
            )

        iterable = []
        size = 0
        for inputs in inputs_loader:
            size += inputs.shape[0]
            iterable.append(lax.map(lambda key: _sample(key, inputs), keys))
        iterable = TargetsLoader.from_iterable(iterable=iterable)
        if return_size:
            return iterable, size
        return iterable

    def mean(
        self,
        inputs_loader: InputsLoader,
        n_posterior_samples: int = 30,
        rng: Optional[PRNGKeyArray] = None,
    ) -> jnp.ndarray:
        r"""
        Estimate the predictive mean of the target variable, that is

        .. math::
            \mathbb{E}_{Y|x, \mathcal{D}}[Y],

        where:
         - :math:`x` is an observed input variable;
         - :math:`Y` is a random target variable;
         - :math:`\mathcal{D}` is the observed training data set;
         - :math:`W` denotes the random model parameters.

        Parameters
        ----------
        inputs_loader : InputsLoader
            A loader of input data points.
        n_posterior_samples : int
            Number of samples to draw from the posterior distribution for each input.
        rng: Optional[PRNGKeyArray]
            A random number generator. If not passed, this will be taken from the attributes of this class.

        Returns
        -------
        jnp.ndarray
            An estimate of the predictive mean for each input.
        """
        if rng is None:
            rng = self.rng.get()
        keys = random.split(rng, n_posterior_samples)

        def fun(i, _curr_sum):
            _sample = self.posterior.sample(inputs_loader=inputs_loader, rng=keys[i])
            _curr_sum += self.likelihood.mean(
                _sample.params,
                inputs_loader,
                _sample.mutable,
                calib_params=_sample.calib_params,
                calib_mutable=_sample.calib_mutable,
            )
            return _curr_sum

        curr_sum = fun(0, 0.0)
        curr_sum = lax.fori_loop(1, n_posterior_samples, fun, curr_sum)
        return curr_sum / n_posterior_samples

    @abc.abstractmethod
    def mode(
        self,
        inputs_loader: InputsLoader,
        n_posterior_samples: int = 30,
        means: Optional[jnp.ndarray] = None,
        rng: Optional[PRNGKeyArray] = None,
    ) -> jnp.ndarray:
        r"""
        Estimate the predictive mode of the target variable, that is

        .. math::
            \text{argmax}_y\ p(y|x, \mathcal{D}),

        where:
         - :math:`x` is an observed input variable;
         - :math:`\mathcal{D}` is the observed training data set;
         - :math:`y` is the target variable to optimize upon.

        Parameters
        ----------
        inputs_loader : InputsLoader
            A loader of input data points.
        n_posterior_samples : int
            Number of samples to draw from the posterior distribution for each input.
        means : Optional[jnp.ndarray] = None
            An estimate of the predictive mean.
        rng : Optional[PRNGKeyArray]
            A random number generator. If not passed, this will be taken from the attributes of this class.

        Returns
        -------
        jnp.ndarray
            An estimate of the predictive mode for each input.
        """
        pass

    def aleatoric_variance(
        self,
        inputs_loader: InputsLoader,
        n_posterior_samples: int = 30,
        rng: Optional[PRNGKeyArray] = None,
    ) -> jnp.ndarray:
        r"""
        Estimate the predictive aleatoric variance of the target variable, that is

        .. math::
            \text{Var}_{W|\mathcal{D}}[\mathbb{E}_{Y|W, x}[Y]],

        where:
         - :math:`x` is an observed input variable;
         - :math:`Y` is a random target variable;
         - :math:`\mathcal{D}` is the observed training data set;
         - :math:`W` denotes the random model parameters.

        Parameters
        ----------
        inputs_loader : InputsLoader
            A loader of input data points.
        n_posterior_samples : int
            Number of samples to draw from the posterior distribution for each input.
        rng : Optional[PRNGKeyArray]
            A random number generator. If not passed, this will be taken from the attributes of this class.

        Returns
        -------
        jnp.ndarray
            An estimate of the predictive aleatoric variance for each input.
        """
        if rng is None:
            rng = self.rng.get()
        keys = random.split(rng, n_posterior_samples)

        def fun(i, _curr_sum):
            _sample = self.posterior.sample(inputs_loader=inputs_loader, rng=keys[i])
            _curr_sum += self.likelihood.variance(
                _sample.params,
                inputs_loader,
                _sample.mutable,
                calib_params=_sample.calib_params,
                calib_mutable=_sample.calib_mutable,
            )
            return _curr_sum

        curr_sum = fun(0, 0.0)
        curr_sum = lax.fori_loop(1, n_posterior_samples, fun, curr_sum)
        return curr_sum / n_posterior_samples

    def epistemic_variance(
        self,
        inputs_loader: InputsLoader,
        n_posterior_samples: int = 30,
        rng: Optional[PRNGKeyArray] = None,
    ) -> jnp.ndarray:
        r"""
        Estimate the predictive epistemic variance of the one-hot encoded target variable, that is

        .. math::
            \mathbb{E}_{W|D}[\text{Var}_{Y|W, x}[Y]],

        where:
         - :math:`x` is an observed input variable;
         - :math:`Y` is a random target variable;
         - :math:`\mathcal{D}` is the observed training data set;
         - :math:`W` denotes the random model parameters.

        Parameters
        ----------
        inputs_loader : InputsLoader
            A loader of input data points.
        n_posterior_samples : int
            Number of samples to draw from the posterior distribution for each input.
        rng : Optional[PRNGKeyArray]
            A random number generator. If not passed, this will be taken from the attributes of this class.

        Returns
        -------
        jnp.ndarray
            An estimate of the predictive epistemic variance for each input.
        """
        if rng is None:
            rng = self.rng.get()
        keys = random.split(rng, n_posterior_samples)

        def fun(i, variables):
            _curr_sum, _curr_sum_sq = variables
            _sample = self.posterior.sample(inputs_loader=inputs_loader, rng=keys[i])
            mean = self.likelihood.mean(
                _sample.params,
                inputs_loader,
                _sample.mutable,
                calib_params=_sample.calib_params,
                calib_mutable=_sample.calib_mutable,
            )
            _curr_sum += mean
            _curr_sum_sq += mean ** 2
            return _curr_sum, _curr_sum_sq

        curr_sum, curr_sum_sq = fun(0, (0.0, 0.0))
        curr_sum, curr_sum_sq = lax.fori_loop(
            1, n_posterior_samples, fun, (curr_sum, curr_sum_sq)
        )
        return jnp.maximum(
            0, curr_sum_sq / n_posterior_samples - (curr_sum / n_posterior_samples) ** 2
        )

    def variance(
        self,
        inputs_loader: InputsLoader,
        n_posterior_samples: int = 30,
        aleatoric_variances: Optional[jnp.ndarray] = None,
        epistemic_variances: Optional[jnp.ndarray] = None,
        rng: Optional[PRNGKeyArray] = None,
    ) -> jnp.ndarray:
        r"""
        Estimate the predictive variance of the target variable, that is

        .. math::
            \text{Var}_{Y|x, D}[Y],

        where:
         - :math:`x` is an observed input variable;
         - :math:`Y` is a random target variable;
         - :math:`\mathcal{D}` is the observed training data set.

        Note that the predictive variance above corresponds to the sum of its aleatoric and epistemic components.

        Parameters
        ----------
        inputs_loader : InputsLoader
            A loader of input data points.
        n_posterior_samples : int
            Number of samples to draw from the posterior distribution for each input.
        aleatoric_variances: Optional[jnp.ndarray]
            An estimate of the aleatoric predictive variance for each input.
        epistemic_variances: Optional[jnp.ndarray]
            An estimate of the epistemic predictive variance for each input.
        rng : Optional[PRNGKeyArray]
            A random number generator. If not passed, this will be taken from the attributes of this class.

        Returns
        -------
        jnp.ndarray
            An estimate of the predictive variance for each input.
        """
        if rng is None:
            rng = self.rng.get()
        if aleatoric_variances is None:
            rng, key = random.split(rng)
            aleatoric_variances = self.aleatoric_variance(
                inputs_loader=inputs_loader,
                n_posterior_samples=n_posterior_samples,
                rng=key,
            )
        if epistemic_variances is None:
            rng, key = random.split(rng)
            epistemic_variances = self.epistemic_variance(
                inputs_loader=inputs_loader,
                n_posterior_samples=n_posterior_samples,
                rng=key,
            )
        return aleatoric_variances + epistemic_variances

    def std(
        self,
        inputs_loader: InputsLoader,
        n_posterior_samples: int = 30,
        variances: Optional[jnp.ndarray] = None,
        rng: Optional[PRNGKeyArray] = None,
    ) -> jnp.ndarray:
        r"""
        Estimate the predictive standard deviation of the target variable, that is

        .. math::
            \text{Var}_{Y|x, D}[Y],

        where:
         - :math:`x` is an observed input variable;
         - :math:`Y` is a random target variable;
         - :math:`\mathcal{D}` is the observed training data set.

        Parameters
        ----------
        inputs_loader : InputsLoader
            A loader of input data points.
        n_posterior_samples : int
            Number of samples to draw from the posterior distribution for each input.
        variances: Optional[jnp.ndarray]
            An estimate of the predictive variance.
        rng : Optional[PRNGKeyArray]
            A random number generator. If not passed, this will be taken from the attributes of this class.

        Returns
        -------
        jnp.ndarray
            An estimate of the predictive standard deviation for each input.
        """
        if variances is None:
            variances = self.variance(
                inputs_loader=inputs_loader,
                n_posterior_samples=n_posterior_samples,
                rng=rng,
            )
        return jnp.sqrt(variances)
