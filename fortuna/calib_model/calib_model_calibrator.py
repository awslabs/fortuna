from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax.numpy as jnp
from flax.core import FrozenDict
from jax._src.prng import PRNGKeyArray
from optax._src.base import PyTree

from fortuna.calib_model.state import CalibState
from fortuna.training.trainer import JittedMixin, MultiDeviceMixin, TrainerABC
from fortuna.typing import Array, Batch, CalibMutable, CalibParams, Mutable, Params


class CalibModelCalibrator(TrainerABC):
    def training_loss_step(
        self,
        loss_fun: Callable[[Any], Union[float, Tuple[float, dict]]],
        params: Params,
        batch: Batch,
        mutable: Mutable,
        rng: PRNGKeyArray,
        n_data: int,
        unravel: Optional[Callable[[any], PyTree]] = None,
        calib_params: Optional[CalibParams] = None,
        calib_mutable: Optional[CalibMutable] = None,
        kwargs: FrozenDict[str, Any] = FrozenDict(),
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        return_aux = ["outputs"]
        if mutable is not None:
            return_aux += ["mutable"]
        loss, aux = loss_fun(
            params,
            batch,
            n_data=n_data,
            mutable=mutable,
            return_aux=return_aux,
            train=True,
            rng=rng,
            calib_params=calib_params,
            calib_mutable=calib_mutable,
        )
        logging_kwargs = None
        return (
            loss,
            {
                "outputs": aux.get("outputs"),
                "mutable": aux.get("mutable"),
                "logging_kwargs": logging_kwargs,
            },
        )

    def validation_step(
        self,
        state: CalibState,
        batch: Batch,
        loss_fun: Callable[[Any], Union[float, Tuple[float, dict]]],
        rng: PRNGKeyArray,
        n_data: int,
        metrics: Optional[Tuple[Callable[[jnp.ndarray, Array], float], ...]] = None,
        unravel: Optional[Callable[[any], PyTree]] = None,
        kwargs: FrozenDict[str, Any] = FrozenDict(),
    ) -> Dict[str, jnp.ndarray]:
        loss, aux = loss_fun(
            state.params,
            batch,
            n_data=n_data,
            mutable=state.mutable,
            return_aux=["outputs"],
            train=False,
            calib_params=state.calib_params,
            calib_mutable=state.calib_mutable,
        )

        if metrics is not None:
            val_metrics = self.compute_metrics(
                self.predict_fn(aux["outputs"]),
                batch[1],
                metrics,
                self.uncertainty_fn(aux["outputs"]),
            )
            return {
                "val_loss": loss,
                **{f"val_{m}": v for m, v in val_metrics.items()},
            }
        return dict(val_loss=loss)


class JittedCalibModelCalibrator(JittedMixin, CalibModelCalibrator):
    pass


class MultiDeviceCalibModelCalibrator(MultiDeviceMixin, CalibModelCalibrator):
    pass
