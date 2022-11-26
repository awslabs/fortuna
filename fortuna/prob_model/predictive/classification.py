from typing import Optional

import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, vmap
from jax._src.prng import PRNGKeyArray

from fortuna.data.loader import InputsLoader
from fortuna.prob_model.posterior.base import Posterior
from fortuna.prob_model.predictive.base import Predictive


class ClassificationPredictive(Predictive):
    def __init__(self, posterior: Posterior):
        """
        Classification predictive distribution class.

        Parameters
        ----------
        posterior : Posterior
             A posterior distribution object.
        """
        super().__init__(posterior)

    def mean(
        self,
        inputs_loader: InputsLoader,
        n_posterior_samples: int = 30,
        rng: Optional[PRNGKeyArray] = None,
    ) -> jnp.ndarray:
        r"""
        Estimate the predictive mean of the one-hot encoded target variable, that is

        .. math::
            \mathbb{E}_{\tilde{Y}|x, \mathcal{D}}[\tilde{Y}],

        where:
         - :math:`x` is an observed input variable;
         - :math:`\tilde{Y}` is a one-hot encoded random target variable;
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
        return super().mean(inputs_loader, n_posterior_samples, rng)

    def mode(
        self,
        inputs_loader: InputsLoader,
        n_posterior_samples: int = 30,
        means: Optional[jnp.ndarray] = None,
        rng: Optional[PRNGKeyArray] = None,
    ) -> jnp.ndarray:
        if means is None:
            means = self.mean(
                inputs_loader=inputs_loader,
                n_posterior_samples=n_posterior_samples,
                rng=rng,
            )
        return jnp.argmax(means, -1)

    def aleatoric_variance(
        self,
        inputs_loader: InputsLoader,
        n_posterior_samples: int = 30,
        rng: Optional[PRNGKeyArray] = None,
    ) -> jnp.ndarray:
        r"""
        Estimate the predictive aleatoric variance of the one-hot encoded target variable, that is

        .. math::
            \text{Var}_{W|\mathcal{D}}[\mathbb{E}_{\tilde{Y}|W, x}[\tilde{Y}]],

        where:
         - :math:`x` is an observed input variable;
         - :math:`\tilde{Y}` is a one-hot encoded random target variable;
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
        return super().aleatoric_variance(inputs_loader, n_posterior_samples, rng)

    def epistemic_variance(
        self,
        inputs_loader: InputsLoader,
        n_posterior_samples: int = 30,
        rng: Optional[PRNGKeyArray] = None,
    ) -> jnp.ndarray:
        r"""
        Estimate the predictive epistemic variance of the one-hot encoded target variable, that is

        .. math::
            \mathbb{E}_{W|D}[\text{Var}_{\tilde{Y}|W, x}[\tilde{Y}]],

        where:
         - :math:`x` is an observed input variable;
         - :math:`\tilde{Y}` is a one-hot encoded random target variable;
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
        return super().epistemic_variance(inputs_loader, n_posterior_samples, rng)

    def variance(
        self,
        inputs_loader: InputsLoader,
        n_posterior_samples: int = 30,
        aleatoric_variances: Optional[jnp.ndarray] = None,
        epistemic_variances: Optional[jnp.ndarray] = None,
        rng: Optional[PRNGKeyArray] = None,
    ) -> jnp.ndarray:
        r"""
        Estimate the predictive variance of the one-hot encoded target variable, that is

        .. math::
            \text{Var}_{\tilde{Y}|x, D}[\tilde{Y}],

        where:
         - :math:`x` is an observed input variable;
         - :math:`\tilde{Y}` is a one-hot encoded random target variable;
         - :math:`\mathcal{D}` is the observed training data set.

        Parameters
        ----------
        inputs_loader : InputsLoader
            A loader of input data points.
        n_posterior_samples : int
            Number of samples to draw from the posterior distribution for each input.
        aleatoric_variances: Optional[jnp.ndarray]
            An estimate of the aleatoric predictive variance.
        epistemic_variances: Optional[jnp.ndarray]
            An estimate of the epistemic predictive variance.
        rng : Optional[PRNGKeyArray]
            A random number generator. If not passed, this will be taken from the attributes of this class.

        Returns
        -------
        jnp.ndarray
            An estimate of the predictive variance for each input.
        """
        return super().variance(
            inputs_loader,
            n_posterior_samples,
            aleatoric_variances,
            epistemic_variances,
            rng,
        )

    def std(
        self,
        inputs_loader: InputsLoader,
        n_posterior_samples: int = 30,
        variances: Optional[jnp.ndarray] = None,
        rng: Optional[PRNGKeyArray] = None,
    ) -> jnp.ndarray:
        r"""
        Estimate the predictive standard deviation of the one-hot encoded target variable, that is

        .. math::
            \sqrt{\text{Var}_{\tilde{Y}|x, D}[\tilde{Y}]},

        where:
         - :math:`x` is an observed input variable;
         - :math:`\tilde{Y}` is a one-hot encoded random target variable;
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
        return super().std(inputs_loader, n_posterior_samples, variances, rng)

    def aleatoric_entropy(
        self,
        inputs_loader: InputsLoader,
        n_posterior_samples: int = 30,
        rng: Optional[PRNGKeyArray] = None,
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
        n_posterior_samples : int
            Number of samples to draw from the posterior distribution for each input.
        rng : Optional[PRNGKeyArray]
            A random number generator. If not passed, this will be taken from the attributes of this class.

        Returns
        -------
        jnp.ndarray
            An estimate of the predictive aleatoric entropy for each input.
        """
        ensemble_outputs = self._sample_calibrated_outputs(
            inputs_loader=inputs_loader, n_output_samples=n_posterior_samples, rng=rng
        )
        n_classes = ensemble_outputs.shape[-1]

        @jit
        def _entropy_term(i: int):
            targets = i * jnp.ones(ensemble_outputs.shape[1])

            def _log_lik_fun(outputs):
                return self.likelihood.prob_output_layer.log_prob(outputs, targets)

            log_liks = vmap(_log_lik_fun)(ensemble_outputs)
            return jnp.mean(jnp.exp(log_liks) * log_liks, 0)

        return -jnp.sum(vmap(_entropy_term)(jnp.arange(n_classes)), 0)

    def epistemic_entropy(
        self,
        inputs_loader: InputsLoader,
        n_posterior_samples: int = 30,
        rng: Optional[PRNGKeyArray] = None,
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
        rng : Optional[PRNGKeyArray]
            A random number generator. If not passed, this will be taken from the attributes of this class.

        Returns
        -------
        jnp.ndarray
            An estimate of the predictive epistemic entropy for each input.
        """
        ensemble_outputs = self._sample_calibrated_outputs(
            inputs_loader=inputs_loader, n_output_samples=n_posterior_samples, rng=rng
        )
        n_classes = ensemble_outputs.shape[-1]

        @jit
        def _entropy_term(i: int):
            targets = i * jnp.ones(ensemble_outputs.shape[1])

            def _log_lik_fun(outputs):
                return self.likelihood.prob_output_layer.log_prob(outputs, targets)

            log_liks = vmap(_log_lik_fun)(ensemble_outputs)
            log_preds = jsp.special.logsumexp(log_liks, 0) - jnp.log(
                n_posterior_samples
            )
            return jnp.exp(log_preds) * log_preds - jnp.mean(
                jnp.exp(log_liks) * log_liks, 0
            )

        return -jnp.sum(vmap(_entropy_term)(jnp.arange(n_classes)), 0)

    def entropy(
        self,
        inputs_loader: InputsLoader,
        n_posterior_samples: int = 30,
        rng: Optional[PRNGKeyArray] = None,
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
        n_posterior_samples : int
            Number of samples to draw from the posterior distribution for each input.
        rng : Optional[PRNGKeyArray]
            A random number generator. If not passed, this will be taken from the attributes of this class.

        Returns
        -------
        jnp.ndarray
            An estimate of the predictive entropy for each input.
        """
        ensemble_outputs = self._sample_calibrated_outputs(
            inputs_loader=inputs_loader, n_output_samples=n_posterior_samples, rng=rng
        )
        n_classes = ensemble_outputs.shape[-1]

        @jit
        def _entropy_term(i: int):
            targets = i * jnp.ones(ensemble_outputs.shape[1])

            def _log_lik_fun(outputs):
                return self.likelihood.prob_output_layer.log_prob(outputs, targets)

            log_liks = vmap(_log_lik_fun)(ensemble_outputs)
            log_preds = jsp.special.logsumexp(log_liks, 0) - jnp.log(
                n_posterior_samples
            )
            return jnp.exp(log_preds) * log_preds

        return -jnp.sum(vmap(_entropy_term)(jnp.arange(n_classes)), 0)
