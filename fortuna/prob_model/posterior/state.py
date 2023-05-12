from __future__ import annotations

from typing import (
    Any,
    Dict,
    Optional,
)

import jax.numpy as jnp
from flax.core import FrozenDict
from flax.training import dynamic_scale

from fortuna.training.train_state import TrainState
from fortuna.typing import (
    CalibMutable,
    CalibParams,
    Mutable,
    OptaxOptimizer,
    Params,
)
from fortuna.utils.strings import convert_string_to_jnp_array


class PosteriorState(TrainState):
    """
    A posterior distribution state. This includes all the parameters and mutable objects that characterize an
    approximation of the posterior distribution.
    """

    params: Params
    mutable: Optional[Mutable] = None
    calib_params: Optional[CalibParams] = None
    calib_mutable: Optional[CalibMutable] = None
    grad_accumulated: Optional[jnp.ndarray] = None
    dynamic_scale: Optional[dynamic_scale.DynamicScale] = None
    encoded_name: jnp.ndarray = convert_string_to_jnp_array("PosteriorState")

    @classmethod
    def init(
        cls,
        params: Params,
        mutable: Optional[Mutable] = None,
        optimizer: Optional[OptaxOptimizer] = None,
        calib_params: Optional[CalibParams] = None,
        calib_mutable: Optional[CalibMutable] = None,
        grad_accumulated: Optional[jnp.ndarray] = None,
        dynamic_scale: Optional[dynamic_scale.DynamicScale] = None,
        **kwargs,
    ) -> Any:
        """
        Initialize a posterior distribution state.

        Parameters
        ----------
        params : Params
            The parameters characterizing an approximation of the posterior distribution.
        optimizer : Optional[OptaxOptimizer]
            An Optax optimizer associated with the posterior state.
        mutable : Optional[Mutable]
            The mutable objects characterizing an approximation of the posterior distribution.
        calib_params : Optional[CalibParams]
            The parameters objects characterizing an approximation of the posterior distribution.
        calib_mutable : Optional[CalibMutable]
            The calibration mutable objects characterizing an approximation of the posterior distribution.
        grad_accumulated : Optional[jnp.ndarray]
            The gradients accumulated in consecutive training steps (used only when `gradient_accumulation_steps > 1`).
        dynamic_scale: Optional[dynamic_scale.DynamicScale]
            Dynamic loss scaling for mixed precision gradients.
        Returns
        -------
        Any
            A posterior distribution state.
        """
        return cls(
            apply_fn=None,
            params=params,
            opt_state=kwargs["opt_state"]
            if optimizer is None and "opt_state" in kwargs
            else optimizer.init(params),
            mutable=mutable,
            step=kwargs.get("step", 0),
            tx=optimizer,
            calib_params=calib_params,
            calib_mutable=calib_mutable,
            dynamic_scale=dynamic_scale,
            grad_accumulated=grad_accumulated,
            **{
                k: v
                for k, v in kwargs.items()
                if k not in ["opt_state", "apply_fn", "tx", "step"]
            },
        )

    @classmethod
    def init_from_dict(
        cls,
        d: Dict,
        optimizer: Optional[OptaxOptimizer] = None,
        **kwargs,
    ) -> PosteriorState:
        """
        Initialize a posterior distribution state from a dictionary.

        Parameters
        ----------
        d : Dict
            A dictionary including attributes of the posterior state.
        optimizer : Optional[OptaxOptimizer]
            An optax optimizer to assign to the posterior state.

        Returns
        -------
        PosteriorState
            A posterior state.
        """
        kwargs = {
            **kwargs,
            **{
                k: v
                for k, v in d.items()
                if k
                not in [
                    "params",
                    "mutable",
                    "optimizer",
                    "calib_params",
                    "calib_mutable",
                ]
            },
        }
        return cls.init(
            FrozenDict(d["params"]),
            FrozenDict(d["mutable"]) if d["mutable"] is not None else None,
            optimizer,
            FrozenDict(d.get("calib_params"))
            if d["calib_params"] is not None
            else None,
            FrozenDict(d.get("calib_mutable"))
            if d["calib_mutable"] is not None
            else None,
            **kwargs,
        )
