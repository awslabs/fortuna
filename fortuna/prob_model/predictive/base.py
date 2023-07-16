import abc
import inspect
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from jax import (
    jit,
    lax,
    random,
)
from jax._src.prng import PRNGKeyArray
from jax.experimental.pjit import pjit
import jax.numpy as jnp
import jax.scipy as jsp
from jax.sharding import PartitionSpec
from jax.tree_util import tree_map

from fortuna.data.loader import (
    DataLoader,
    InputsLoader,
    TargetsLoader,
)
from fortuna.data.loader.base import ShardedPrefetchedLoader
from fortuna.prob_model.posterior.base import Posterior
from fortuna.typing import (
    Array,
    Batch,
    CalibMutable,
    CalibParams,
)
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
        rng: Optional[PRNGKeyArray] = None,
        shard: bool = True,
        **kwargs,
    ) -> jnp.ndarray:
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
            that would be produced using the posterior distribution state.
        rng : Optional[PRNGKeyArray]
            A random number generator. If not passed, this will be taken from the attributes of this class.
        shard: bool
            Whether to shard computation over multiple devices, if available.

        Returns
        -------
        jnp.ndarray
            An estimate of the predictive log-pdf for each data point.
        """
        if rng is None:
            rng = self.rng.get()

        return self._loop_fun_through_loader(
            self._batched_log_prob,
            data_loader,
            n_posterior_samples,
            rng,
            shard,
            **kwargs,
        )

    def _batched_log_prob(
        self,
        batch: Batch,
        n_posterior_samples: int = 30,
        rng: Optional[PRNGKeyArray] = None,
        shard: bool = True,
        **kwargs,
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, dict]]:
        if rng is None:
            rng = self.rng.get()
        keys = random.split(rng, n_posterior_samples)

        def _lik_log_batched_prob(params, mutable, calib_params, calib_mutable):
            return self.likelihood._batched_log_prob(
                params,
                batch,
                mutable=mutable,
                calib_params=calib_params,
                calib_mutable=calib_mutable,
                **kwargs,
            )

        if shard and self.posterior.partition_manager.shardings is not None:
            _lik_log_batched_prob = pjit(
                _lik_log_batched_prob,
                in_shardings=(
                    self.posterior.partition_manager.shardings.params,
                    self.posterior.partition_manager.shardings.mutable,
                    self.posterior.partition_manager.shardings.calib_params,
                    self.posterior.partition_manager.shardings.calib_mutable,
                ),
                out_shardings=PartitionSpec(("dp", "fsdp")),
            )
        else:
            _lik_log_batched_prob = jit(_lik_log_batched_prob)

        def _fun(key):
            sample = self.posterior.sample(inputs=batch[0], rng=key)
            with self.posterior.partition_manager.partitioner.mesh:
                return _lik_log_batched_prob(
                    sample.params,
                    sample.mutable,
                    sample.calib_params,
                    sample.calib_mutable,
                )

        return jsp.special.logsumexp(
            jnp.stack(list(map(_fun, keys))), axis=0
        ) - jnp.log(n_posterior_samples)

    def _batched_log_joint_prob(
        self,
        batch: Batch,
        n_data: int,
        n_posterior_samples: int = 30,
        return_aux: Optional[List[str]] = None,
        ensemble_outputs: Optional[jnp.ndarray] = None,
        calib_params: Optional[CalibParams] = None,
        calib_mutable: Optional[CalibMutable] = None,
        rng: Optional[PRNGKeyArray] = None,
        **kwargs,
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Dict]]:
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

            def _lik_log_joint_prob(key):
                sample = self.posterior.sample(inputs=batch[0], rng=key)
                return self.likelihood._batched_log_joint_prob(
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
                    **kwargs,
                )

            outs = lax.map(_lik_log_joint_prob, keys)
        else:
            if calib_params is None:
                d = self.posterior.state.extract_calib_keys()
                calib_params = d["calib_params"]
                calib_mutable = d["calib_mutable"]

            def _lik_log_joint_prob(outputs):
                return self.likelihood._batched_log_joint_prob(
                    None,
                    batch,
                    n_data,
                    mutable=None,
                    outputs=outputs,
                    return_aux=return_aux,
                    calib_params=calib_params,
                    calib_mutable=calib_mutable,
                    **kwargs,
                )

            outs = lax.map(_lik_log_joint_prob, ensemble_outputs)

        if len(return_aux) > 0:
            ensemble_log_lik, aux = outs
            aux = tree_map(lambda v: jnp.mean(v, 0), aux)
        else:
            ensemble_log_lik = outs
        log_pred = jsp.special.logsumexp(ensemble_log_lik) - jnp.log(
            len(ensemble_log_lik)
        )
        if len(return_aux) > 0:
            return log_pred, aux
        return log_pred

    def _batched_negative_log_joint_prob(
        self,
        batch: Batch,
        n_data: int,
        n_posterior_samples: int = 30,
        return_aux: Optional[List[str]] = None,
        ensemble_outputs: Optional[jnp.ndarray] = None,
        calib_params: Optional[CalibParams] = None,
        calib_mutable: Optional[CalibMutable] = None,
        rng: Optional[PRNGKeyArray] = None,
        **kwargs,
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Dict]]:
        outs = self._batched_log_joint_prob(
            batch,
            n_data,
            n_posterior_samples,
            return_aux,
            ensemble_outputs,
            calib_params,
            calib_mutable,
            rng,
            **kwargs,
        )
        if len(return_aux) > 0:
            loss, aux = outs
            loss *= -1
            return loss, aux
        return -outs

    def sample(
        self,
        inputs_loader: InputsLoader,
        n_target_samples: int = 1,
        return_aux: Optional[List[str]] = None,
        rng: Optional[PRNGKeyArray] = None,
        shard: bool = True,
        **kwargs,
    ) -> Union[Tuple[Array, Dict[str, Array]], Array]:
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
        **kwargs
        inputs_loader : InputsLoader
            A loader of input data points.
        n_target_samples : int
            Number of target samples to sample for each input data point.
        return_aux : Optional[List[str]]
            Return auxiliary objects. We currently support 'outputs'.
        rng : Optional[PRNGKeyArray]
            A random number generator. If not passed, this will be taken from the attributes of this class.
        shard: bool
            Whether to shard computation over multiple devices, if available.

        Returns
        -------
        Tuple[Array, Dict[str, Array]] | Array
            Samples for each input data point. Optionally, an auxiliary object is returned.
        """
        if not rng:
            rng = self.rng.get()

        return self._loop_fun_through_loader(
            self._batched_sample,
            inputs_loader,
            n_target_samples,
            rng,
            shard,
            is_fun_ensembled=False,
            return_aux=return_aux,
            **kwargs,
        )

    def _batched_sample(
        self,
        inputs: Array,
        n_target_samples: int = 1,
        return_aux: Optional[List[str]] = None,
        rng: Optional[PRNGKeyArray] = None,
        shard: bool = True,
        **kwargs,
    ) -> Union[jnp.ndarray, Dict[str, jnp.ndarray]]:
        if return_aux is None:
            return_aux = []

        if rng is None:
            rng = self.rng.get()
        keys = random.split(rng, n_target_samples)

        def _sample(rng, params, mutable, calib_params, calib_mutable):
            return self.likelihood._batched_sample(
                1,
                params,
                inputs,
                mutable=mutable,
                calib_params=calib_params,
                calib_mutable=calib_mutable,
                return_aux=return_aux,
                rng=rng,
                **kwargs,
            )

        if shard and self.posterior.partition_manager.shardings is not None:
            _sample = pjit(
                _sample,
                in_shardings=(
                    self.posterior.partition_manager.shardings.params,
                    self.posterior.partition_manager.shardings.mutable,
                    self.posterior.partition_manager.shardings.calib_params,
                    self.posterior.partition_manager.shardings.calib_mutable,
                ),
                out_shardings=PartitionSpec(("dp", "fsdp"))
                if not len(return_aux)
                else (PartitionSpec(("dp", "fsdp")), PartitionSpec()),
            )

        def _fun(key):
            key1, key2 = random.split(key, 2)
            with self.posterior.partition_manager.partitioner.mesh:
                sample = self.posterior.sample(inputs=inputs, rng=key1)
                return _sample(
                    key2,
                    sample.params,
                    sample.mutable,
                    sample.calib_params,
                    sample.calib_mutable,
                )

        samples = list(map(_fun, keys))
        if len(return_aux):
            samples, aux = samples
            return jnp.stack(samples), aux
        return jnp.stack(samples)

    def sample_calibrated_outputs(
        self,
        inputs_loader: InputsLoader,
        n_output_samples: int = 1,
        rng: Optional[PRNGKeyArray] = None,
        shard: bool = True,
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
        shard: bool
            Whether to shard computation over multiple devices, if available.

        Returns
        -------
        jnp.ndarray
            Samples of calibrated outputs.
        """
        if rng is None:
            rng = self.rng.get()

        return self._loop_fun_through_loader(
            self._sample_batched_calibrated_outputs,
            inputs_loader,
            n_output_samples,
            rng,
            shard,
            is_fun_ensembled=True,
        )

    def _sample_batched_calibrated_outputs(
        self,
        inputs: Array,
        n_output_samples: int = 1,
        rng: Optional[PRNGKeyArray] = None,
        shard: bool = True,
    ) -> jnp.ndarray:
        if rng is None:
            rng = self.rng.get()
        keys = random.split(rng, n_output_samples)

        def _apply_fn(params, mutable, calib_params, calib_mutable):
            return self.likelihood._get_batched_calibrated_outputs(
                params=params,
                inputs=inputs,
                mutable=mutable,
                calib_params=calib_params,
                calib_mutable=calib_mutable,
            )

        if shard and self.posterior.partition_manager.shardings is not None:
            _apply_fn = pjit(
                _apply_fn,
                in_shardings=(
                    self.posterior.partition_manager.shardings.params,
                    self.posterior.partition_manager.shardings.mutable,
                    self.posterior.partition_manager.shardings.calib_params,
                    self.posterior.partition_manager.shardings.calib_mutable,
                ),
                out_shardings=PartitionSpec(("fsdp", "dp")),
            )
        else:
            _apply_fn = jit(_apply_fn)

        def _sample(key):
            sample = self.posterior.sample(inputs=inputs, rng=key)
            with self.posterior.partition_manager.partitioner.mesh:
                return _apply_fn(sample.params, sample.mutable)

        return jnp.stack(list(map(_sample, keys)))

    def _sample_outputs(
        self,
        inputs_loader: InputsLoader,
        n_output_samples: int = 1,
        rng: Optional[PRNGKeyArray] = None,
        shard: bool = True,
    ) -> jnp.ndarray:
        if rng is None:
            rng = self.rng.get()

        return self._loop_fun_through_loader(
            self._sample_batched_outputs,
            inputs_loader,
            n_output_samples,
            rng,
            shard,
        )

    def _sample_batched_outputs(
        self,
        inputs: Array,
        n_output_samples: int = 1,
        rng: Optional[PRNGKeyArray] = None,
        shard: bool = True,
    ) -> jnp.ndarray:
        if rng is None:
            rng = self.rng.get()
        keys = random.split(rng, n_output_samples)

        def _apply_fn(params, mutable):
            return self.likelihood.model_manager.apply(
                params=params, inputs=inputs, mutable=mutable
            )

        if shard and getattr(self.posterior.partition_manager, "shardings") is not None:
            _apply_fn = pjit(
                _apply_fn,
                in_shardings=(
                    self.posterior.partition_manager.shardings.params,
                    self.posterior.partition_manager.shardings.mutable,
                ),
                out_shardings=PartitionSpec(("fsdp", "dp")),
            )
        else:
            _apply_fn = jit(_apply_fn)

        def _sample(key):
            sample = self.posterior.sample(inputs=inputs, rng=key)
            with self.posterior.partition_manager.partitioner.mesh:
                return _apply_fn(sample.params, sample.mutable)

        return jnp.stack(list(map(_sample, keys)))

    def _sample_outputs_loader(
        self,
        inputs_loader: InputsLoader,
        n_output_samples: int = 1,
        rng: Optional[PRNGKeyArray] = None,
        return_size: bool = False,
        shard: bool = True,
    ) -> Union[TargetsLoader, Tuple[TargetsLoader, int]]:
        if rng is None:
            rng = self.rng.get()
        keys = random.split(rng, n_output_samples)

        iterable = []
        size = 0
        for inputs in inputs_loader:
            size += (
                inputs.shape[0]
                if not isinstance(inputs, dict)
                else inputs[list(inputs.keys())[0]].shape[0]
            )
            outputs = jnp.stack(
                list(
                    map(
                        lambda key: self._sample_batched_outputs(
                            inputs=inputs,
                            rng=key,
                            shard=shard,
                        )[0],
                        keys,
                    )
                )
            )
            iterable.append(outputs)
        iterable = TargetsLoader.from_iterable(iterable=iterable)
        if return_size:
            return iterable, size
        return iterable

    def mean(
        self,
        inputs_loader: InputsLoader,
        n_posterior_samples: int = 30,
        rng: Optional[PRNGKeyArray] = None,
        shard: bool = True,
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
        shard: bool
            Whether to shard computation over multiple devices, if available.

        Returns
        -------
        jnp.ndarray
            An estimate of the predictive mean for each input.
        """
        if rng is None:
            rng = self.rng.get()

        return self._loop_fun_through_loader(
            self._batched_mean, inputs_loader, n_posterior_samples, rng, shard
        )

    def _batched_mean(
        self,
        inputs: Array,
        n_posterior_samples: int = 30,
        rng: Optional[PRNGKeyArray] = None,
        shard: bool = True,
    ) -> jnp.ndarray:
        if rng is None:
            rng = self.rng.get()
        keys = random.split(rng, n_posterior_samples)

        def _lik_batched_mean(params, mutable, calib_params, calib_mutable):
            return self.likelihood._batched_mean(
                params,
                inputs,
                mutable,
                calib_params=calib_params,
                calib_mutable=calib_mutable,
            )

        if shard and self.posterior.partition_manager.shardings is not None:
            _lik_batched_mean = pjit(
                _lik_batched_mean,
                in_shardings=(
                    self.posterior.partition_manager.shardings.params,
                    self.posterior.partition_manager.shardings.mutable,
                    self.posterior.partition_manager.shardings.calib_params,
                    self.posterior.partition_manager.shardings.calib_mutable,
                ),
                out_shardings=PartitionSpec(("dp", "fsdp")),
            )
        else:
            _lik_batched_mean = jit(_lik_batched_mean)

        def fun(key):
            _sample = self.posterior.sample(inputs=inputs, rng=key)
            with self.posterior.partition_manager.partitioner.mesh:
                return _lik_batched_mean(
                    params=_sample.params,
                    mutable=_sample.mutable,
                    calib_params=_sample.calib_params,
                    calib_mutable=_sample.calib_mutable,
                )

        return jnp.mean(list(map(fun, keys)), axis=0)

    @abc.abstractmethod
    def mode(
        self,
        inputs_loader: InputsLoader,
        n_posterior_samples: int = 30,
        means: Optional[jnp.ndarray] = None,
        rng: Optional[PRNGKeyArray] = None,
        shard: bool = True,
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
        shard: bool
            Whether to shard computation over multiple devices, if available.

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
        shard: bool = True,
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
        shard: bool
            Whether to shard computation over multiple devices, if available.

        Returns
        -------
        jnp.ndarray
            An estimate of the predictive aleatoric variance for each input.
        """
        if rng is None:
            rng = self.rng.get()

        return self._loop_fun_through_inputs_loader(
            self._batched_aleatoric_variance,
            inputs_loader,
            n_posterior_samples,
            rng,
            shard,
        )

    def _batched_aleatoric_variance(
        self,
        inputs: Array,
        n_posterior_samples: int = 30,
        rng: Optional[PRNGKeyArray] = None,
    ) -> jnp.ndarray:
        if rng is None:
            rng = self.rng.get()
        keys = random.split(rng, n_posterior_samples)

        def fun(i, _curr_sum):
            _sample = self.posterior.sample(inputs=inputs, rng=keys[i])
            _curr_sum += self.likelihood._batched_variance(
                _sample.params,
                inputs,
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
        shard: bool = True,
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
        shard: bool
            Whether to shard computation over multiple devices, if available.

        Returns
        -------
        jnp.ndarray
            An estimate of the predictive epistemic variance for each input.
        """
        if rng is None:
            rng = self.rng.get()

        return self._loop_fun_through_loader(
            self._batched_epistemic_variance,
            inputs_loader,
            n_posterior_samples,
            rng,
            shard,
        )

    def _batched_epistemic_variance(
        self,
        inputs: Array,
        n_posterior_samples: int = 30,
        rng: Optional[PRNGKeyArray] = None,
    ) -> jnp.ndarray:
        if rng is None:
            rng = self.rng.get()
        keys = random.split(rng, n_posterior_samples)

        def fun(i, variables):
            _curr_sum, _curr_sum_sq = variables
            _sample = self.posterior.sample(inputs=inputs, rng=keys[i])
            mean = self.likelihood._batched_mean(
                _sample.params,
                inputs,
                _sample.mutable,
                calib_params=_sample.calib_params,
                calib_mutable=_sample.calib_mutable,
            )
            _curr_sum += mean
            _curr_sum_sq += mean**2
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
        shard: bool = True,
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
        shard: bool
            Whether to shard computation over multiple devices, if available.

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
                shard=shard,
            )
        if epistemic_variances is None:
            rng, key = random.split(rng)
            epistemic_variances = self.epistemic_variance(
                inputs_loader=inputs_loader,
                n_posterior_samples=n_posterior_samples,
                rng=key,
                shard=shard,
            )
        return aleatoric_variances + epistemic_variances

    def std(
        self,
        inputs_loader: InputsLoader,
        n_posterior_samples: int = 30,
        variances: Optional[jnp.ndarray] = None,
        rng: Optional[PRNGKeyArray] = None,
        shard: bool = True,
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
        shard: bool
            Whether to shard computation over multiple devices, if available.

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
                shard=shard,
            )
        return jnp.sqrt(variances)

    def _loop_fun_through_loader(
        self,
        fun: Callable,
        loader: Union[InputsLoader, DataLoader, TargetsLoader],
        n_posterior_samples: int,
        rng: PRNGKeyArray,
        shard: bool,
        is_fun_ensembled: bool = False,
        return_aux: Optional[List[str]] = None,
        **kwargs,
    ) -> Union[tuple[Any, ...], Array]:
        if shard and self.posterior.partition_manager.shardings is not None:
            loader = ShardedPrefetchedLoader(
                loader=loader, partition_manager=self.posterior.partition_manager
            )

        def fun2(_data):
            if "return_aux" in inspect.getfullargspec(fun)[0]:
                return fun(
                    _data,
                    n_posterior_samples,
                    rng,
                    shard,
                    return_aux=return_aux,
                    **kwargs,
                )
            return fun(_data, n_posterior_samples, rng, shard, **kwargs)

        outs = [fun2(data) for data in loader]
        if return_aux is not None:
            return tuple(
                [
                    tree_map(lambda v: jnp.concatenate(v, int(is_fun_ensembled)), out)
                    for out in zip(*outs)
                ]
            )
        return jnp.concatenate(outs, int(is_fun_ensembled))
