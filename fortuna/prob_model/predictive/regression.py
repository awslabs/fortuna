from typing import (
    List,
    Optional,
    Union,
)

from jax import (
    lax,
    random,
    vmap,
)
from jax._src.prng import PRNGKeyArray
import jax.numpy as jnp
import jax.scipy as jsp

from fortuna.data.loader import InputsLoader
from fortuna.prob_model.posterior.base import Posterior
from fortuna.prob_model.predictive.base import Predictive
from fortuna.typing import Array


class RegressionPredictive(Predictive):
    def __init__(self, posterior: Posterior):
        """
        Regression predictive distribution class.

        Parameters
        ----------
        posterior : Posterior
             A posterior distribution object.
        """
        super().__init__(posterior)

    def mode(
        self,
        inputs_loader: InputsLoader,
        n_posterior_samples: int = 30,
        means: Optional[jnp.ndarray] = None,
        rng: Optional[PRNGKeyArray] = None,
        distribute: bool = True,
    ) -> jnp.ndarray:
        if means is not None:
            return means
        return self.mean(
            inputs_loader=inputs_loader,
            n_posterior_samples=n_posterior_samples,
            rng=rng,
            distribute=distribute,
        )

    def aleatoric_entropy(
        self,
        inputs_loader: InputsLoader,
        n_posterior_samples: int = 30,
        n_target_samples: int = 30,
        rng: Optional[PRNGKeyArray] = None,
        distribute: bool = True,
    ) -> jnp.ndarray:
        r"""
        Estimate the predictive aleatoric entropy, that is

        .. math::
            -\mathbb{E}_{W|\mathcal{D}}[\mathbb{E}_{Y|W, x}[\log p(Y|W, x)]],

        where:
         - :math:`x` is an observed input variable;
         - :math:`Y` is a random target variable;
         - :math:`\mathcal{D}` is the observed training data set;
         - :math:`W` denotes the random model parameters.

        Parameters
        ----------
        inputs_loader : InputsLoader
            A loader of input data points.
        n_target_samples: int
            Number of target samples to draw for each input.
        n_posterior_samples : int
            Number of samples to draw from the posterior distribution for each input.
        rng : Optional[PRNGKeyArray]
            A random number generator. If not passed, this will be taken from the attributes of this class.
        distribute: bool
            Whether to distribute computation over multiple devices, if available.

        Returns
        -------
        jnp.ndarray
            An estimate of the predictive aleatoric entropy for each input.
        """
        if rng is None:
            rng = self.rng.get()
        key1, *keys = random.split(rng, 1 + n_posterior_samples)

        ensemble_outputs = self.sample_calibrated_outputs(
            inputs_loader=inputs_loader,
            n_output_samples=n_posterior_samples,
            rng=key1,
            distribute=distribute,
        )

        ensemble_target_samples = lax.map(
            lambda variables: self.likelihood.prob_output_layer.sample(
                n_target_samples, variables[0], rng=variables[1]
            ),
            (ensemble_outputs, jnp.array(keys)),
        )

        def fun(i, _curr_sum):
            log_liks = self.likelihood.prob_output_layer.log_prob(
                ensemble_outputs[i], ensemble_target_samples[i]
            )
            _curr_sum -= jnp.mean(log_liks, 0)
            return _curr_sum

        curr_sum = fun(0, 0.0)
        curr_sum = lax.fori_loop(1, n_posterior_samples, fun, curr_sum)
        return curr_sum / n_posterior_samples

    def epistemic_entropy(
        self,
        inputs_loader: InputsLoader,
        n_posterior_samples: int = 30,
        n_target_samples: int = 30,
        rng: Optional[PRNGKeyArray] = None,
        distribute: bool = True,
    ) -> jnp.ndarray:
        r"""
        Estimate the predictive epistemic entropy, that is

        .. math::
            -\mathbb{E}_{Y|x, \mathcal{D}}[\log p(Y|x, \mathcal{D})] +
            \mathbb{E}_{W|\mathcal{D}}[\mathbb{E}_{Y|W, x}[\log p(Y|W, x)]],

        where:
         - :math:`x` is an observed input variable;
         - :math:`Y` is a random target variable;
         - :math:`\mathcal{D}` is the observed training data set;
         - :math:`W` denotes the random model parameters.

         Note that the epistemic entropy above is defined as the difference between the predictive entropy and the
         aleatoric predictive entropy.

        Parameters
        ----------
        inputs_loader : InputsLoader
            A loader of input data points.
        n_posterior_samples : int
            Number of samples to draw from the posterior distribution for each input.
        n_target_samples: int
            Number of target samples to draw for each input.
        rng : Optional[PRNGKeyArray]
            A random number generator. If not passed, this will be taken from the attributes of this class.
        distribute: bool
            Whether to distribute computation over multiple devices, if available.

        Returns
        -------
        jnp.ndarray
            An estimate of the predictive epistemic entropy for each input.
        """
        if rng is None:
            rng = self.rng.get()
        key1, *keys = random.split(rng, 1 + n_posterior_samples)

        ensemble_outputs = self.sample_calibrated_outputs(
            inputs_loader=inputs_loader,
            n_output_samples=n_posterior_samples,
            rng=key1,
            distribute=distribute,
        )

        ensemble_target_samples = lax.map(
            lambda variables: self.likelihood.prob_output_layer.sample(
                n_target_samples, variables[0], rng=variables[1]
            ),
            (ensemble_outputs, jnp.array(keys)),
        )

        def fun(i, _curr_sum):
            @vmap
            def _log_pred_fun(target_sample: jnp.ndarray):
                logps = self.likelihood.prob_output_layer.log_prob(
                    ensemble_outputs, target_sample
                )
                return jsp.special.logsumexp(logps, 0) - jnp.log(n_posterior_samples)

            log_preds = _log_pred_fun(ensemble_target_samples[i])
            log_liks = self.likelihood.prob_output_layer.log_prob(
                ensemble_outputs[i], ensemble_target_samples[i]
            )
            _curr_sum -= jnp.mean(log_preds - log_liks, 0)
            return _curr_sum

        curr_sum = fun(0, 0.0)
        curr_sum = lax.fori_loop(1, n_posterior_samples, fun, curr_sum)
        return curr_sum / n_posterior_samples

    def entropy(
        self,
        inputs_loader: InputsLoader,
        n_posterior_samples: int = 30,
        n_target_samples: int = 30,
        rng: Optional[PRNGKeyArray] = None,
        distribute: bool = True,
    ) -> jnp.ndarray:
        r"""
        Estimate the predictive entropy, that is

        .. math::
            -\mathbb{E}_{Y|x, \mathcal{D}}[\log p(Y|x, \mathcal{D})],

        where:
         - :math:`x` is an observed input variable;
         - :math:`Y` is a random target variable;
         - :math:`\mathcal{D}` is the observed training data set;
         - :math:`W` denotes the random model parameters.

        Parameters
        ----------
        inputs_loader : InputsLoader
            A loader of input data points.
        n_target_samples: int
            Number of target samples to draw for each input.
        n_posterior_samples : int
            Number of samples to draw from the posterior distribution for each input.
        rng : Optional[PRNGKeyArray]
            A random number generator. If not passed, this will be taken from the attributes of this class.
        distribute: bool
            Whether to distribute computation over multiple devices, if available.

        Returns
        -------
        jnp.ndarray
            An estimate of the predictive entropy for each input.
        """
        if rng is None:
            rng = self.rng.get()
        key1, *keys = random.split(rng, 1 + n_posterior_samples)

        ensemble_outputs = self.sample_calibrated_outputs(
            inputs_loader=inputs_loader,
            n_output_samples=n_posterior_samples,
            rng=key1,
            distribute=distribute,
        )

        ensemble_target_samples = lax.map(
            lambda variables: self.likelihood.prob_output_layer.sample(
                n_target_samples, variables[0], rng=variables[1]
            ),
            (ensemble_outputs, jnp.array(keys)),
        )

        def fun(i, _curr_sum):
            @vmap
            def _log_pred_fun(target_sample: jnp.ndarray):
                logps = self.likelihood.prob_output_layer.log_prob(
                    ensemble_outputs, target_sample
                )
                return jsp.special.logsumexp(logps, 0) - jnp.log(n_posterior_samples)

            log_preds = _log_pred_fun(ensemble_target_samples[i])
            _curr_sum -= jnp.mean(log_preds, 0)
            return _curr_sum

        curr_sum = fun(0, 0.0)
        curr_sum = lax.fori_loop(1, n_posterior_samples, fun, curr_sum)
        return curr_sum / n_posterior_samples

    def credible_interval(
        self,
        inputs_loader: InputsLoader,
        n_target_samples: int = 30,
        error: float = 0.05,
        interval_type: str = "two-tailed",
        rng: Optional[PRNGKeyArray] = None,
        distribute: bool = True,
    ) -> jnp.ndarray:
        r"""
        Estimate credible intervals for the target variable. This is supported only if the target variable is scalar.

        Parameters
        ----------
        inputs_loader : InputsLoader
            A loader of input data points.
        n_target_samples: int
            Number of target samples to draw for each input.
        error: float
            The interval error. This must be a number between 0 and 1, extremes included. For example,
            `error=0.05` corresponds to a 95% level of credibility.
        interval_type: str
            The interval type. We support "two-tailed" (default), "right-tailed" and "left-tailed".
        rng : Optional[PRNGKeyArray]
            A random number generator. If not passed, this will be taken from the attributes of this class.
        distribute: bool
            Whether to distribute computation over multiple devices, if available.

        Returns
        -------
        jnp.ndarray
            A credibility interval for each of the inputs.
        """
        supported_types = ["two-tailed", "right-tailed", "left-tailed"]
        if interval_type not in supported_types:
            raise ValueError(
                "`type={}` not recognised. Please choose among the following supported types: {}.".format(
                    interval_type, supported_types
                )
            )
        q = (
            jnp.array([0.5 * error, 1 - 0.5 * error])
            if interval_type == "two-tailed"
            else error if interval_type == "left-tailed" else 1 - error
        )
        qq = self.quantile(
            q=q,
            inputs_loader=inputs_loader,
            n_target_samples=n_target_samples,
            rng=rng,
            distribute=distribute,
        )
        if qq.shape[-1] != 1:
            raise ValueError(
                """Credibility intervals are only supported for scalar target variables."""
            )
        if interval_type == "two-tailed":
            lq, uq = qq.squeeze(2)
            return jnp.array(list(zip(lq, uq)))
        else:
            return qq

    def quantile(
        self,
        q: Union[float, Array, List],
        inputs_loader: InputsLoader,
        n_target_samples: Optional[int] = 30,
        rng: Optional[PRNGKeyArray] = None,
        distribute: bool = True,
    ) -> Union[float, jnp.ndarray]:
        r"""
        Estimate the `q`-th quantiles of the predictive probability density function.

        Parameters
        ----------
        q : Union[float, Array, List]
            Quantile or sequence of quantiles to compute. Each of these must be between 0 and 1, extremes included.
        inputs_loader : InputsLoader
            A loader of input data points.
        n_target_samples : int
            Number of target samples to sample for each input data point.
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
        if type(q) == list:
            q = jnp.array(q)
        samples = self.sample(
            inputs_loader=inputs_loader,
            n_target_samples=n_target_samples,
            rng=rng,
            distribute=distribute,
        )
        return jnp.quantile(samples, q, axis=0)
