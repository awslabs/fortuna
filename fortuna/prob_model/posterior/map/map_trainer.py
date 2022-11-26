from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax.numpy as jnp
from flax.core import FrozenDict
from jax._src.prng import PRNGKeyArray
from optax._src.base import PyTree

from fortuna.prob_model.posterior.map import *
from fortuna.prob_model.posterior.map.map_state import MAPState
from fortuna.prob_model.posterior.posterior_trainer import PosteriorTrainerABC
from fortuna.training.trainer import JittedMixin, MultiGPUMixin
from fortuna.typing import (Array, Batch, CalibMutable, CalibParams, Mutable,
                            Params)


class MAPTrainer(PosteriorTrainerABC):
    def training_loss_step(
        self,
        fun: Callable[[Any], Union[float, Tuple[float, dict]]],
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
        loss, aux = fun(
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
        loss = -loss
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
        state: MAPState,
        batch: Batch,
        fun: Callable[[Any], Union[float, Tuple[float, dict]]],
        rng: PRNGKeyArray,
        n_data: int,
        metrics: Optional[Tuple[Callable[[jnp.ndarray, Array], float], ...]] = None,
        unravel: Optional[Callable[[any], PyTree]] = None,
        kwargs: FrozenDict[str, Any] = FrozenDict(),
    ) -> Dict[str, jnp.ndarray]:
        log_probabilities, aux = fun(
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
                self.predict_fn(aux["outputs"]), batch[1], metrics
            )
            return {
                "val_loss": -log_probabilities,
                **{f"val_{m}": v for m, v in val_metrics.items()},
            }
        return dict(val_loss=-log_probabilities)

    def __str__(self):
        return MAP_NAME


class JittedMAPTrainer(JittedMixin, MAPTrainer):
    pass


class MultiGPUMAPTrainer(MultiGPUMixin, MAPTrainer):
    pass
