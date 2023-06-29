from typing import Optional

from jax import (
    jit,
    vmap,
)
from jax._src.prng import PRNGKeyArray
import jax.numpy as jnp
import jax.scipy as jsp

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
        distribute: bool = True,
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
        distribute: bool
            Whether to distribute computation over multiple devices, if available.

        Returns
        -------
        jnp.ndarray
            An estimate of the predictive mean for each input.
        """
        return super().mean(inputs_loader, n_posterior_samples, rng, distribute)

    def mode(
        self,
        inputs_loader: InputsLoader,
        n_posterior_samples: int = 30,
        means: Optional[jnp.ndarray] = None,
        rng: Optional[PRNGKeyArray] = None,
        distribute: bool = True,
    ) -> jnp.ndarray:
        if means is None:
            means = self.mean(
                inputs_loader=inputs_loader,
                n_posterior_samples=n_posterior_samples,
                rng=rng,
                distribute=distribute,
            )
        return jnp.argmax(means, -1)

    def aleatoric_variance(
        self,
        inputs_loader: InputsLoader,
        n_posterior_samples: int = 30,
        rng: Optional[PRNGKeyArray] = None,
        distribute: bool = True,
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
        distribute: bool
            Whether to distribute computation over multiple devices, if available.

        Returns
        -------
        jnp.ndarray
            An estimate of the predictive aleatoric variance for each input.
        """
        return super().aleatoric_variance(
            inputs_loader, n_posterior_samples, rng, distribute
        )

    def epistemic_variance(
        self,
        inputs_loader: InputsLoader,
        n_posterior_samples: int = 30,
        rng: Optional[PRNGKeyArray] = None,
        distribute: bool = True,
        **kwargs,
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
        distribute: bool
            Whether to distribute computation over multiple devices, if available.

        Returns
        -------
        jnp.ndarray
            An estimate of the predictive epistemic variance for each input.
        """
        return super().epistemic_variance(
            inputs_loader, n_posterior_samples, rng, distribute
        )

    def variance(
        self,
        inputs_loader: InputsLoader,
        n_posterior_samples: int = 30,
        aleatoric_variances: Optional[jnp.ndarray] = None,
        epistemic_variances: Optional[jnp.ndarray] = None,
        rng: Optional[PRNGKeyArray] = None,
        distribute: bool = True,
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
        distribute: bool
            Whether to distribute computation over multiple devices, if available.

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
            distribute,
        )

    def std(
        self,
        inputs_loader: InputsLoader,
        n_posterior_samples: int = 30,
        variances: Optional[jnp.ndarray] = None,
        rng: Optional[PRNGKeyArray] = None,
        distribute: bool = True,
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
        distribute: bool
            Whether to distribute computation over multiple devices, if available.

        Returns
        -------
        jnp.ndarray
            An estimate of the predictive standard deviation for each input.
        """
        return super().std(
            inputs_loader, n_posterior_samples, variances, rng, distribute
        )

    def aleatoric_entropy(
        self,
        inputs_loader: InputsLoader,
        n_posterior_samples: int = 30,
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
        ensemble_outputs = self.sample_calibrated_outputs(
            inputs_loader=inputs_loader,
            n_output_samples=n_posterior_samples,
            rng=rng,
            distribute=distribute,
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
        rng : Optional[PRNGKeyArray]
            A random number generator. If not passed, this will be taken from the attributes of this class.
        distribute: bool
            Whether to distribute computation over multiple devices, if available.

        Returns
        -------
        jnp.ndarray
            An estimate of the predictive epistemic entropy for each input.
        """
        ensemble_outputs = self.sample_calibrated_outputs(
            inputs_loader=inputs_loader,
            n_output_samples=n_posterior_samples,
            rng=rng,
            distribute=distribute,
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
        ensemble_outputs = self.sample_calibrated_outputs(
            inputs_loader=inputs_loader,
            n_output_samples=n_posterior_samples,
            rng=rng,
            distribute=distribute,
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
    
    def conformal_set(
        self,
        train_test_data_loader: InputsLoader,
        n: int,
        n_test: int,
        n_posterior_samples: int = 30,
        error: float = 0.05,
        rng: Optional[PRNGKeyArray] = None,
        ESS: bool = False,
    ) -> jnp.ndarray:
        r"""
        Estimate conformal sets for the target variable. 

        Parameters
        ----------
        train_test_data_loader : InputsLoader
            A loader of input data points where the first n data are training, 
            followed by n_test with Y = 0 then n_test with Y = 1.
        n : int
            Size of training set
        n_test : int
            Size of test set
        n_posterior_samples : int
            Number of samples to draw from the posterior distribution for each input.
        error: float
            The interval error. This must be a number between 0 and 1, extremes included. For example,
            `error=0.05` corresponds to a 95% level of confidence.
        rng : Optional[PRNGKeyArray]
            A random number generator. If not passed, this will be taken from the attributes of this class.
        ESS: bool
            Whether to compute ESS of importance weights or not.

        Returns
        -------
        jnp.ndarray
            A conformal set for each of the inputs.
        """     
        #returns n_posterior_samples x (n_test + n_test +n)
        ensemble_train_test_log_probs = self.ensemble_log_prob(data_loader=train_test_data_loader, n_posterior_samples = n_posterior_samples)

        #Divide back out into test set grid and training
        ensemble_train_log_probs = ensemble_train_test_log_probs[:,0:n] #training log likelihood
        ensemble_test0_log_probs = ensemble_train_test_log_probs[:,n:n + n_test] #test log likelihoods at Y = 0
        ensemble_test1_log_probs= ensemble_train_test_log_probs[:,n+n_test:n+2*n_test] #test log likelihoods at Y = 1

        #log likelihood values for each test input, posterior sample, and grid point (shape =  n_test x  n_posterior_samples x 2)
        ensemble_testgrid_log_probs = (jnp.concatenate((ensemble_test0_log_probs.T.reshape(-1,n_posterior_samples,1),
                                                         ensemble_test1_log_probs.T.reshape(-1,n_posterior_samples,1)),axis = 2))


        @jit #compute rank of nonconformity score (unnormalized by n+1)
        def _compute_cb_region_IS(ensemble_testgrid_log_probs):
            n= jnp.shape(ensemble_train_log_probs)[1] #ensemble_train_log_probs is n_posterior_samples x n; training data log likelihood values
            n_plot = jnp.shape(ensemble_testgrid_log_probs)[1] #ensemble_testgrid_log_probs is n_posterior_samples x n_plot_grid;  test data log likelihood values for each posterior sample
            rank_cp = jnp.zeros(n_plot)

            #compute importance sampling weights and normalizing
            wjk = jnp.exp(ensemble_testgrid_log_probs.T)
            Zjk = jnp.sum(wjk,axis = 1).reshape(-1,1)

            #compute predictives for y_i,x_i and y_new,x_n+1
            p_cp = jnp.dot(wjk/Zjk, jnp.exp(ensemble_train_log_probs))
            p_new = jnp.sum(wjk**2,axis = 1).reshape(-1,1)/Zjk

            #compute nonconformity score and sort
            pred_tot = jnp.concatenate((p_cp,p_new),axis = 1)
            rank_cp = jnp.sum(pred_tot <= pred_tot[:,-1].reshape(-1,1),axis = 1)

            #Compute region of grid which is in confidence set
            region_true =rank_cp> error*(n+1)
            return region_true

        #Compute CB interval for each
        conf_set = vmap(_compute_cb_region_IS)(ensemble_testgrid_log_probs)

        if ESS:
          ## DIAGNOSE IMPORTANCE WEIGHTS ##
          @jit #compute Effective sample size
          def _diagnose_is_weights(ensemble_testgrid_log_probs):
              n= jnp.shape(ensemble_train_log_probs)[1] #ensemble_train_log_probs is n_posterior_samples x n
              n_plot = jnp.shape(ensemble_testgrid_log_probs)[1] #ensemble_testgrid_log_probs is n_posterior_samples x n_plot_grid
              rank_cp = jnp.zeros(n_plot)

              #compute importance sampling weights and normalizing
              logwjk = ensemble_testgrid_log_probs.T.reshape(n_plot,-1, 1)
              logZjk = jsp.special.logsumexp(logwjk,axis = 1)

              #compute predictives for y_i,x_i and y_new,x_n+1
              logp_new = jsp.special.logsumexp(2*logwjk,axis = 1)-logZjk

              #compute ESS
              wjk = jnp.exp(logwjk - logZjk.reshape(-1,1,1))
              ESS = 1/jnp.sum(wjk**2,axis = 1)
              return ESS
          ###
          ESS = vmap(_diagnose_is_weights)(ensemble_testgrid_log_probs)
          return conf_set, ESS
          
        else: return conf_set
