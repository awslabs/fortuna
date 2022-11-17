from __future__ import annotations

from typing import Any, Dict, Optional, Union

import jax.numpy as jnp
from flax.core import FrozenDict
from fortuna.training.train_state import TrainState
from fortuna.typing import Mutable, OptaxOptimizer, Params
from fortuna.utils.strings import convert_string_to_jnp_array


class CalibState(TrainState):
    params: Params
    mutable: Optional[Mutable] = None
    encoded_name: jnp.ndarray = convert_string_to_jnp_array("CalibState")

    @classmethod
    def init(
        cls,
        params: Params,
        mutable: Optional[Mutable] = None,
        optimizer: Optional[OptaxOptimizer] = None,
        **kwargs,
    ) -> Any:
        """
        Initialize a calibration state.

        Parameters
        ----------
        params : Params
            The calibration parameters.
        optimizer : Optional[OptaxOptimizer]
            An Optax optimizer associated with the calibration state.
        mutable : Optional[Mutable]
            The calibration mutable objects.

        Returns
        -------
        Any
            A calibration state.
        """
        return cls(
            apply_fn=None,
            params=params,
            opt_state=kwargs["opt_state"]
            if optimizer is None and "opt_state" in kwargs
            else None
            if optimizer is None
            else optimizer.init(params),
            mutable=mutable,
            step=kwargs.get("step", 0),
            tx=optimizer,
            **{
                k: v
                for k, v in kwargs.items()
                if k not in ["opt_state", "apply_fn", "tx", "step"]
            },
        )

    @classmethod
    def init_from_dict(
        cls,
        d: Union[Dict, FrozenDict],
        optimizer: Optional[OptaxOptimizer] = None,
        **kwargs,
    ) -> CalibState:
        """
        Initialize a calibration state from a dictionary.

        Parameters
        ----------
        d : Union[Dict, FrozenDict]
            A dictionary with as keys the calibrators and as values their initializations.
        optimizer : Optional[OptaxOptimizer]
            An optax optimizer to assign to the calibration state.

        Returns
        -------
        CalibState
            A calibration state.
        """
        return cls.init(
            params=FrozenDict(d["params"]),
            mutable=FrozenDict(d["mutable"]),
            optimizer=optimizer,
            **kwargs,
        )
