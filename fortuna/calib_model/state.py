from __future__ import annotations

from typing import Any, Dict, Optional

from flax.core import FrozenDict

from fortuna.training.train_state import TrainState
from fortuna.typing import CalibMutable, CalibParams, Mutable, OptaxOptimizer, Params


class CalibState(TrainState):
    params: Params
    mutable: Optional[Mutable] = None
    calib_params: Optional[CalibParams] = None
    calib_mutable: Optional[CalibMutable] = None

    @classmethod
    def init(
        cls,
        params: Params,
        mutable: Optional[Mutable] = None,
        optimizer: Optional[OptaxOptimizer] = None,
        calib_params: Optional[CalibParams] = None,
        calib_mutable: Optional[CalibMutable] = None,
        **kwargs,
    ) -> Any:
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
            **{
                k: v
                for k, v in kwargs.items()
                if k not in ["opt_state", "apply_fn", "tx", "step"] and hasattr(cls, k)
            },
        )

    @classmethod
    def init_from_dict(
        cls,
        d: Dict,
        optimizer: Optional[OptaxOptimizer] = None,
        **kwargs,
    ) -> CalibState:
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
