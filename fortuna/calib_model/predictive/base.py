from typing import Optional

from jax._src.prng import PRNGKeyArray
import jax.numpy as jnp

from fortuna.data.loader import (
    DataLoader,
    InputsLoader,
)
from fortuna.likelihood.base import Likelihood
from fortuna.utils.random import WithRNG


class Predictive(WithRNG):
    def __init__(self, likelihood: Likelihood):
        """
        Predictive distribution abstract class.

        Parameters
        ----------
        likelihood : Likelihood
             A likelihood object.
        """
        self.likelihood = likelihood
        self.state = None

    def log_prob(self, data_loader: DataLoader, distribute: bool = True) -> jnp.ndarray:
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
        distribute: bool
            Whether to distribute computation over multiple devices, if available.

        Returns
        -------
        jnp.ndarray
            An estimate of the predictive log-pdf for each data point.
        """
        state = self.state.get()
        return self.likelihood.log_prob(
            params=state.params,
            data_loader=data_loader,
            mutable=state.mutable,
            distribute=distribute,
        )

    def sample(
        self,
        inputs_loader: InputsLoader,
        n_samples: int = 1,
        rng: Optional[PRNGKeyArray] = None,
        distribute: bool = True,
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
        n_samples : int
            Number of target samples to sample for each input data point.
        rng : Optional[PRNGKeyArray]
            A random number generator. If not passed, this will be taken from the attributes of this class.
        distribute: bool
            Whether to distribute computation over multiple devices, if available.

        Returns
        -------
        jnp.ndarray
            Samples for each input data point.
        """
        state = self.state.get()
        return self.likelihood.sample(
            n_target_samples=n_samples,
            params=state.params,
            inputs_loader=inputs_loader,
            mutable=state.mutable,
            rng=rng,
            distribute=distribute,
        )

    def mean(
        self,
        inputs_loader: InputsLoader,
        distribute: bool = True,
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
        distribute: bool
            Whether to distribute computation over multiple devices, if available.

        Returns
        -------
        jnp.ndarray
            An estimate of the predictive mean for each input.
        """
        state = self.state.get()
        return self.likelihood.mean(
            params=state.params,
            inputs_loader=inputs_loader,
            mutable=state.mutable,
            distribute=distribute,
        )

    def mode(
        self,
        inputs_loader: InputsLoader,
        distribute: bool = True,
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
        distribute: bool
            Whether to distribute computation over multiple devices, if available.

        Returns
        -------
        jnp.ndarray
            An estimate of the predictive mode for each input.
        """
        state = self.state.get()
        return self.likelihood.mode(
            params=state.params,
            inputs_loader=inputs_loader,
            mutable=state.mutable,
            distribute=distribute,
        )

    def variance(
        self,
        inputs_loader: InputsLoader,
        distribute: bool = True,
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
        distribute: bool
            Whether to distribute computation over multiple devices, if available.

        Returns
        -------
        jnp.ndarray
            An estimate of the predictive variance for each input.
        """
        state = self.state.get()
        return self.likelihood.variance(
            params=state.params,
            inputs_loader=inputs_loader,
            mutable=state.mutable,
            distribute=distribute,
        )

    def std(
        self,
        inputs_loader: InputsLoader,
        variances: Optional[jnp.ndarray] = None,
        distribute: bool = True,
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
        variances: Optional[jnp.ndarray]
            An estimate of the predictive variance.passed, this will be taken from the attributes of this class.
        distribute: bool
            Whether to distribute computation over multiple devices, if available.

        Returns
        -------
        jnp.ndarray
            An estimate of the predictive standard deviation for each input.
        """
        if variances is None:
            variances = self.variance(
                inputs_loader=inputs_loader,
                distribute=distribute,
            )
        return jnp.sqrt(variances)
