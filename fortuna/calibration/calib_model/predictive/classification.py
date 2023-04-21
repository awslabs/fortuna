import jax.numpy as jnp

from fortuna.data.loader import InputsLoader
from fortuna.likelihood.classification import ClassificationLikelihood
from fortuna.calibration.calib_model.predictive.base import Predictive
from fortuna.typing import Path


class ClassificationPredictive(Predictive):
    def __init__(self, likelihood: ClassificationLikelihood, restore_checkpoint_path: Path):
        super().__init__(likelihood=likelihood, restore_checkpoint_path=restore_checkpoint_path)

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
