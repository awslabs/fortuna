from __future__ import annotations

from typing import (
    Dict,
    Optional,
    Union,
)

from flax.core import FrozenDict

from fortuna.typing import (
    CalibMutable,
    CalibParams,
)


class OutputCalibManagerState:
    params: CalibParams
    mutable: Optional[CalibMutable] = None

    def __init__(self, params: CalibParams, mutable: Optional[CalibMutable] = None):
        """
        An model manager state class.

        Parameters
        ----------
        params : Params
            The random parameters of the probabilistic model.
        mutable : Optional[Mutable]
            The mutable objects used to evaluate the models.
        """
        self.params = params
        self.mutable = mutable

    @classmethod
    def init_from_dict(cls, d: Union[Dict, FrozenDict]) -> OutputCalibManagerState:
        """
        Initialize an output calibration manager state from a dictionary.

        Parameters
        ----------
        d : Union[Dict, FrozenDict]
            A dictionary with as keys the calibrators and as values their initializations.

        Returns
        -------
        OutputCalibManagerState
            An output calibration manager state.
        """
        params = FrozenDict(
            {
                k: FrozenDict({"params": v["params"] if v else None})
                for k, v in d.items()
            }
        )
        mutable = {calib_name: {} for calib_name in d}
        for name, variables in d.items():
            if variables:
                for var_name, var_obj in variables.items():
                    if var_name != "params":
                        mutable[name][var_name] = var_obj
        mutable = FrozenDict(
            {name: v if len(v) > 0 else None for name, v in mutable.items()}
        )
        return cls(params=params, mutable=mutable)
