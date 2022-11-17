from typing import Optional

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from fortuna.prob_output_layer.base import ProbOutputLayer
from fortuna.typing import Array
from jax import random, vmap
from jax._src.prng import PRNGKeyArray


class ClassificationProbOutputLayer(ProbOutputLayer):
    def __init__(self):
        r"""
        Classification probabilistic output layers class. It characterizes the probability distribution of a target
        variable given a calibrated output logits as a Categorical distribution. That is
        :math:`p(y|\omega)=\text{Categorical}(y|p=\text{softmax}(\omega))`, where :math:`y` denotes a target variable
        and :math:`\omega` a calibrated output.
        """
        super().__init__()

    def log_prob(self, outputs: Array, targets: Array, **kwargs) -> Array:
        n_cats = outputs.shape[-1]
        targets = jax.nn.one_hot(targets, n_cats)
        return jnp.sum(targets * outputs, -1) - jsp.special.logsumexp(outputs, -1)

    def predict(self, outputs: Array, **kwargs) -> jnp.ndarray:
        return jnp.argmax(outputs, -1)

    def sample(
        self,
        n_target_samples: int,
        outputs: Array,
        rng: Optional[PRNGKeyArray] = None,
        **kwargs
    ) -> jnp.ndarray:
        probs = (
            jax.nn.softmax(outputs, -1)
            if "probs" not in kwargs or kwargs["probs"] is None
            else kwargs["probs"]
        )
        n_cats = probs.shape[-1]
        if rng is None:
            rng = self.rng.get()
        keys = random.split(rng, probs.shape[0])
        return vmap(
            lambda key, p: random.choice(key, n_cats, p=p, shape=(n_target_samples,)),
            out_axes=1,
        )(keys, probs)

    def mean(self, outputs: Array, **kwargs) -> jnp.ndarray:
        """
        Estimate the mean of the one-hot encoded target variable given the output with respect to the probabilistic
        output layer distribution.

        Parameters
        ----------
        outputs : Array
            Model outputs

        Returns
        -------
        jnp.ndarray
            The estimated mean for each output.
        """
        return jax.nn.softmax(outputs, -1)

    def mode(self, outputs: Array, **kwargs) -> jnp.ndarray:
        """
        Estimate the mode of the one-hot encoded target variable given the output with respect to the probabilistic
        output layer distribution.

        Parameters
        ----------
        outputs : Array
            Model outputs

        Returns
        -------
        jnp.ndarray
            The estimated mode for each output.
        """
        return jnp.argmax(outputs, -1)

    def variance(self, outputs: Array, **kwargs) -> jnp.ndarray:
        """
        Estimate the variance of the one-hot encoded target variable given the output with respect to the probabilistic
        output layer distribution.

        Parameters
        ----------
        outputs : Array
            Model outputs

        Returns
        -------
        jnp.ndarray
            The estimated variance for each output.
        """
        p = self.mean(outputs)
        return p * (1 - p)

    def std(self, outputs: Array, variances: Optional[Array] = None) -> jnp.ndarray:
        """
        Estimate the standard deviation of the one-hot encoded target variable given the output with respect to the
        probabilistic output layer distribution.

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
        return super().std(outputs, variances)

    def entropy(self, outputs: Array, **kwargs) -> jnp.ndarray:
        """
        Estimate the entropy of the one-hot encoded target variable given the output with respect to the probabilistic
        output layer distribution.

        Parameters
        ----------
        outputs : Array
            Model outputs

        Returns
        -------
        jnp.ndarray
            The estimated entropy for each output.
        """
        n_classes = outputs.shape[-1]

        @vmap
        def _entropy_term(i: int):
            targets = i * jnp.ones(outputs.shape[0])
            log_liks = self.log_prob(outputs, targets, **kwargs)
            return jnp.exp(log_liks) * log_liks

        return -jnp.sum(_entropy_term(jnp.arange(n_classes)), 0)
