from typing import (
    Any,
    Callable,
    Dict,
    Tuple,
    Union,
)

from jax._src.prng import PRNGKeyArray
import jax.numpy as jnp

from fortuna.output_calib_model.state import OutputCalibState
from fortuna.training.output_calibrator.base import OutputCalibratorABC
from fortuna.training.output_calibrator.mixins.sharding import ShardingMixin
from fortuna.typing import (
    Array,
    Batch,
    CalibMutable,
    CalibParams,
)


class ProbModelOutputCalibrator(OutputCalibratorABC):
    def training_loss_step(
        self,
        loss_fun: Callable[[Any], Union[float, Tuple[float, dict]]],
        params: CalibParams,
        batch: Batch,
        outputs: Array,
        mutable: CalibMutable,
        rng: PRNGKeyArray,
        n_data: int,
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        return_aux = ["outputs"]
        if mutable is not None:
            return_aux += ["mutable"]
        loss, aux = loss_fun(
            batch,
            n_data=n_data,
            return_aux=["outputs", "calib_mutable"],
            ensemble_outputs=outputs,
            calib_params=params,
            calib_mutable=mutable,
            rng=rng,
        )
        logging_kwargs = None
        return (
            loss,
            {
                "outputs": aux.get("outputs"),
                "mutable": aux.get("calib_mutable"),
                "logging_kwargs": logging_kwargs,
            },
        )

    def validation_loss_step(
        self,
        state: OutputCalibState,
        batch: Batch,
        outputs: Array,
        loss_fun: Callable,
        rng: PRNGKeyArray,
        n_data: int,
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        loss, aux = loss_fun(
            batch,
            n_data=n_data,
            return_aux=["outputs"],
            ensemble_outputs=outputs,
            calib_params=state.params,
            calib_mutable=state.mutable,
            rng=rng,
        )
        return loss, aux

    def __str__(self):
        return "calibration"


class ShardedProbModelOutputCalibrator(ShardingMixin, ProbModelOutputCalibrator):
    pass
