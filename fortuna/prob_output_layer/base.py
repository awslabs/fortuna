import abc
from typing import Optional

import jax.numpy as jnp
from jax._src.prng import PRNGKeyArray

from fortuna.typing import Array
from fortuna.utils.random import WithRNG


class ProbOutputLayer(WithRNG, abc.ABC):
    r"""
    Abstract probabilistic output layer class. It characterizes the distribution of the target variable given the
    calibrated outputs. It can be see as :math:`p(y|\omega)`, where :math:`y` is a target variable and :math:`\omega` a
    calibrated output. The probabilistic output layer is not join over different data points, and it acts on them
    individually.
    """

    @abc.abstractmethod
    def log_prob(self, outputs: Array, targets: Array, **kwargs) -> jnp.ndarray:
        """
        Evaluate the log-probability density function (a.k.a. log-pdf) of target variables for each of the outputs.

        Parameters
        ----------
        outputs : Array
            Calibrated outputs.
        targets : Array
            Target data points.

        Returns
        -------
        jnp.ndarray
            An evaluation of the log-pdf for each output.
        """
        pass

    @abc.abstractmethod
    def predict(self, outputs: Array, **kwargs) -> jnp.ndarray:
        """
        Predict target variables starting from the calibrated outputs.

        Parameters
        ----------
        outputs : Array
            Calibrated outputs.

        Returns
        -------
        jnp.ndarray
            A predictions for each output.
        """

    @abc.abstractmethod
    def sample(
        self,
        n_target_samples: int,
        outputs: Array,
        rng: Optional[PRNGKeyArray] = None,
        **kwargs,
    ) -> jnp.ndarray:
        """
        Sample target variables for each outputs.

        Parameters
        ----------
        n_target_samples: int
            The number of target samples to draw for each of the outputs.
        outputs : Array
            Calibrated outputs.
        rng : Optional[PRNGKeyArray]
            A random number generator. If not passed, this will be taken from the attributes of this class.

        Returns
        -------
        jnp.ndarray
            Samples of the target variable for each output.
        """
        pass

    @abc.abstractmethod
    def mean(self, outputs: Array, **kwargs) -> jnp.ndarray:
        """
        Estimate the mean of the target variable given the output with respect to the probabilistic output layer
        distribution.

        Parameters
        ----------
        outputs : Array
            Model outputs

        Returns
        -------
        jnp.ndarray
            The estimated mean for each output.
        """
        pass

    @abc.abstractmethod
    def mode(self, outputs: Array, **kwargs) -> jnp.ndarray:
        """
        Estimate the mode of the target variable given the output with respect to the probabilistic output layer
        distribution.

        Parameters
        ----------
        outputs : Array
            Model outputs

        Returns
        -------
        jnp.ndarray
            The estimated mode for each output.
        """
        pass

    @abc.abstractmethod
    def variance(self, outputs: Array, **kwargs) -> jnp.ndarray:
        """
        Estimate the variance of the target variable given the output with respect to the probabilistic output layer
        distribution.

        Parameters
        ----------
        outputs : Array
            Model outputs

        Returns
        -------
        jnp.ndarray
            The estimated variance for each output.
        """
        pass

    def std(self, outputs: Array, variances: Optional[Array] = None) -> jnp.ndarray:
        """
        Estimate the standard deviation of the target variable given the output with respect to the probabilistic
        output layer distribution.

        Parameters
        ----------
        outputs : Array
            Model outputs
        variances: Optional[Array]
            Variance for each output.

        Returns
        -------
        jnp.ndarray
            The estimated standard deviation for each output.
        """
        return jnp.sqrt(self.variance(outputs)) if variances is None else variances

    @abc.abstractmethod
    def entropy(self, outputs: Array, **kwargs) -> jnp.ndarray:
        """
        Estimate the entropy of the target variable given the output with respect to the probabilistic output layer
        distribution.

        Parameters
        ----------
        outputs : Array
            Model outputs

        Returns
        -------
        jnp.ndarray
            The estimated mean for each output.
        """
        pass
