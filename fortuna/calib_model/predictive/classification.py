import jax.numpy as jnp

from fortuna.data.loader import InputsLoader
from fortuna.likelihood.classification import ClassificationLikelihood
from fortuna.calib_model.predictive.base import Predictive


class ClassificationPredictive(Predictive):
    def __init__(self, likelihood: ClassificationLikelihood):
        super().__init__(likelihood=likelihood)

    def entropy(
        self,
        inputs_loader: InputsLoader,
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
            distribute=distribute
        )
