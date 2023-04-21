import jax.numpy as jnp

from fortuna.data.loader import InputsLoader
from fortuna.likelihood.regression import RegressionLikelihood
from fortuna.calibration.calib_model.predictive.base import Predictive
from fortuna.training.train_state_repository import TrainStateRepository
from typing import Union, List, Optional
from fortuna.typing import Array, Path
from jax._src.prng import PRNGKeyArray


class RegressionPredictive(Predictive):
    def __init__(self, likelihood: RegressionLikelihood, restore_checkpoint_path: Path):
        super().__init__(likelihood=likelihood, restore_checkpoint_path=restore_checkpoint_path)

    def entropy(
        self,
        inputs_loader: InputsLoader,
        n_target_samples: int = 30,
        distribute: bool = True,

    ) -> jnp.ndarray:
        state = self.state.get()
        return self.likelihood.entropy(
            params=state.params,
            inputs_loader=inputs_loader,
            mutable=state.mutable,
            distribute=distribute,
            n_target_samples=n_target_samples
        )

    def quantile(
        self,
        q: Union[float, Array, List],
        inputs_loader: InputsLoader,
        n_target_samples: Optional[int] = 30,
        rng: Optional[PRNGKeyArray] = None,
        distribute: bool = True,
    ) -> Union[float, jnp.ndarray]:
        state = self.state.get()
        return self.likelihood.quantile(
            q=q,
            params=state.params,
            inputs_loader=inputs_loader,
            mutable=state.mutable,
            n_target_samples=n_target_samples,
            rng=rng,
            distribute=distribute
        )

    def credible_interval(
        self,
        inputs_loader: InputsLoader,
        n_target_samples: int = 30,
        error: float = 0.05,
        interval_type: str = "two-tailed",
        rng: Optional[PRNGKeyArray] = None,
        distribute: bool = True,
    ) -> jnp.ndarray:
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
            else error
            if interval_type == "left-tailed"
            else 1 - error
        )
        qq = self.quantile(
            q=q,
            inputs_loader=inputs_loader,
            n_target_samples=n_target_samples,
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

