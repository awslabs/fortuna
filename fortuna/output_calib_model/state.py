from __future__ import annotations

from typing import (
    Any,
    Dict,
    Optional,
    Union,
)

from flax.core import FrozenDict

from fortuna.training.train_state import TrainState
from fortuna.typing import (
    CalibMutable,
    CalibParams,
    OptaxOptimizer,
)
from fortuna.utils.strings import convert_string_to_tuple


class OutputCalibState(TrainState):
    params: CalibParams
    mutable: Optional[CalibMutable] = None
    encoded_name: tuple = convert_string_to_tuple("OutputCalibState")

    @classmethod
    def init(
        cls,
        params: CalibParams,
        mutable: Optional[CalibMutable] = None,
        optimizer: Optional[OptaxOptimizer] = None,
        **kwargs,
    ) -> Any:
        """
        Initialize an output calibration state.

        Parameters
        ----------
        params : CalibParams
            The calibration parameters.
        optimizer : Optional[OptaxOptimizer]
            An Optax optimizer associated with the calibration state.
        mutable : Optional[CalibMutable]
            The calibration mutable objects.

        Returns
        -------
        Any
            A calibration state.
        """
        return cls(
            apply_fn=None,
            params=params,
            opt_state=(
                kwargs["opt_state"]
                if optimizer is None and "opt_state" in kwargs
                else None if optimizer is None else optimizer.init(params)
            ),
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
    ) -> OutputCalibState:
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
        OutputCalibState
            A calibration state.
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
                ]
            },
        }
        return cls.init(
            FrozenDict(d["params"]),
            FrozenDict(d["mutable"]),
            optimizer,
            **kwargs,
        )
