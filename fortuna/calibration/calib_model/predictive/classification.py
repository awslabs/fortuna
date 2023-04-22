import jax.numpy as jnp

from fortuna.data.loader import InputsLoader
from fortuna.likelihood.classification import ClassificationLikelihood
from fortuna.calibration.calib_model.predictive.base import Predictive


class ClassificationPredictive(Predictive):
    def __init__(self, likelihood: ClassificationLikelihood):
        super().__init__(likelihood=likelihood)

    def entropy(
        self,
        inputs_loader: InputsLoader,
        distribute: bool = True,
    ) -> jnp.ndarray:
        state = self.state.get()
        return self.likelihood.entropy(
            params=state.params,
            inputs_loader=inputs_loader,
            mutable=state.mutable,
            distribute=distribute
        )
