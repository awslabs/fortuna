from typing import (
    List,
    Optional,
    Union,
)

import jax.numpy as jnp
from jax import (
    random,
    vmap,
)
from jax._src.prng import PRNGKeyArray

from fortuna.prob_output_layer.base import ProbOutputLayer
from fortuna.typing import Array


class RegressionProbOutputLayer(ProbOutputLayer):
    def __init__(self):
        r"""
        Regression probabilistic output layers class. It characterizes the probability distribution of a target
        variable given a calibrated output as a Gaussian distribution. That is
        :math:`p(y|\mu, \sigma^2)=\text{Normal}(y|\mu, \sigma^2)`, where :math:`y` denotes a target variable and
        :math:`\omega=[\mu, \log\sigma^2]` a calibrated output.

        """
        super().__init__()
        self.log2pi = jnp.log(2 * jnp.pi)

    def log_prob(self, outputs: Array, targets: Array, **kwargs) -> jnp.ndarray:
        means, log_vars = jnp.split(outputs, 2, axis=-1)
        return -0.5 * jnp.sum(
            jnp.exp(-log_vars) * (targets - means) ** 2 + log_vars + self.log2pi, -1
        )

    def predict(self, outputs: Array, **kwargs) -> jnp.ndarray:
        return jnp.split(outputs, 2, axis=-1)[0]

    def sample(
        self,
        n_target_samples: int,
        outputs: Array,
        rng: Optional[PRNGKeyArray] = None,
        **kwargs,
    ) -> jnp.ndarray:
        if rng is None:
            rng = self.rng.get()
        means, log_vars = jnp.split(outputs, 2, axis=-1)
        return means + jnp.exp(0.5 * log_vars) * random.normal(
            rng, (n_target_samples,) + means.shape
        )

    def quantile(
        self,
        q: Union[float, Array, List],
        outputs: Array,
        n_target_samples: Optional[int] = 30,
        rng: Optional[PRNGKeyArray] = None,
    ) -> jnp.ndarray:
        if type(q) == list:
            q = jnp.array(q)
        samples = self.sample(
            n_target_samples=n_target_samples, outputs=outputs, rng=rng
        )
        return jnp.quantile(samples, q, axis=0)

    def mean(self, outputs: Array, **kwargs) -> jnp.ndarray:
        return jnp.split(outputs, 2, axis=-1)[0]

    def mode(self, outputs: Array, **kwargs) -> jnp.ndarray:
        return jnp.split(outputs, 2, axis=-1)[0]

    def variance(self, outputs: Array, **kwargs) -> jnp.ndarray:
        return jnp.exp(jnp.split(outputs, 2, axis=-1)[1])

    def entropy(
        self,
        outputs: Array,
        n_target_samples: int = 30,
        rng: Optional[PRNGKeyArray] = None,
        **kwargs,
    ) -> jnp.ndarray:
        samples = self.sample(n_target_samples, outputs, rng=rng, **kwargs)

        @vmap
        def _log_lik_fun(sample: jnp.ndarray):
            return self.log_prob(outputs, sample, **kwargs)

        return -jnp.mean(_log_lik_fun(samples), 0)

    def credible_interval(
        self,
        outputs: Array,
        n_target_samples: int = 30,
        error: float = 0.05,
        interval_type: str = "two-tailed",
        rng: Optional[PRNGKeyArray] = None,
    ) -> jnp.ndarray:
        r"""
        Estimate credible intervals for the target variable. This is supported only if the target variable is scalar.

        Parameters
        ----------
        outputs: Array
            Model outputs.
        n_target_samples: int
            Number of target samples to draw for each output.
        error: float
            The interval error. This must be a number between 0 and 1, extremes included. For example,
            `error=0.05` corresponds to a 95% level of credibility.
        interval_type: str
            The interval type. We support "two-tailed" (default), "right-tailed" and "left-tailed".
        rng : Optional[PRNGKeyArray]
            A random number generator. If not passed, this will be taken from the attributes of this class.

        Returns
        -------
        jnp.ndarray
            A credibility interval for each of the outputs.
        """
        supported_types = ["two-tailed", "right-tailed", "left-tailed"]
        if interval_type not in supported_types:
            raise ValueError(
                "`type={}` not recognised. Please choose among the following supported types: {}.".format(
                    supported_types
                )
            )
        q = (
            jnp.array([0.5 * error, 1 - 0.5 * error])
            if interval_type == "two-tailed"
            else error
            if interval_type == "left-tailed"
            else 1 - error
        )
        qq = self.quantile(
            q=q, outputs=outputs, n_target_samples=n_target_samples, rng=rng
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
