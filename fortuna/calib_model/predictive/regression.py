from typing import (
    List,
    Optional,
    Union,
)

import jax
import jax.numpy as jnp

from fortuna.calib_model.predictive.base import Predictive
from fortuna.data.loader import InputsLoader
from fortuna.likelihood.regression import RegressionLikelihood
from fortuna.typing import Array


class RegressionPredictive(Predictive):
    def __init__(self, likelihood: RegressionLikelihood):
        super().__init__(likelihood=likelihood)

    def entropy(
        self,
        inputs_loader: InputsLoader,
        n_samples: int = 30,
        rng: Optional[jax.Array] = None,
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
        n_samples : int
            Number of samples to draw for each input.
        rng : Optional[jax.Array]
            A random number generator. If not passed, this will be taken from the attributes of this class.
        distribute: bool
            Whether to distribute computation over multiple devices, if available.

        Returns
        -------
        jnp.ndarray
            An estimate of the predictive entropy for each input.
        """
        state = self.state.get()
        return self.likelihood.entropy(
            params=state.params,
            inputs_loader=inputs_loader,
            mutable=state.mutable,
            distribute=distribute,
            n_target_samples=n_samples,
            rng=rng,
        )

    def quantile(
        self,
        q: Union[float, Array, List],
        inputs_loader: InputsLoader,
        n_samples: Optional[int] = 30,
        rng: Optional[jax.Array] = None,
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
        n_samples : int
            Number of target samples to sample for each input data point.
        rng: Optional[jax.Array]
            A random number generator. If not passed, this will be taken from the attributes of this class.
        distribute: bool
            Whether to distribute computation over multiple devices, if available.

        Returns
        -------
        jnp.ndarray
            Quantile estimate for each quantile and each input. If multiple quantiles `q` are given, the result's
            first axis is over different quantiles.
        """
        state = self.state.get()
        return self.likelihood.quantile(
            q=q,
            params=state.params,
            inputs_loader=inputs_loader,
            mutable=state.mutable,
            n_target_samples=n_samples,
            rng=rng,
            distribute=distribute,
        )

    def credible_interval(
        self,
        inputs_loader: InputsLoader,
        n_samples: int = 30,
        error: float = 0.05,
        interval_type: str = "two-tailed",
        rng: Optional[jax.Array] = None,
        distribute: bool = True,
    ) -> jnp.ndarray:
        r"""
        Estimate credible intervals for the target variable. This is supported only if the target variable is scalar.

        Parameters
        ----------
        inputs_loader : InputsLoader
            A loader of input data points.
        n_samples: int
            Number of target samples to draw for each input.
        error: float
            The interval error. This must be a number between 0 and 1, extremes included. For example,
            `error=0.05` corresponds to a 95% level of credibility.
        interval_type: str
            The interval type. We support "two-tailed" (default), "right-tailed" and "left-tailed".
        rng : Optional[jax.Array]
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
            n_samples=n_samples,
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
