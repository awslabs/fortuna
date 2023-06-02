from __future__ import annotations

from typing import (
    Dict,
    Optional,
    Union,
)

from flax.core import FrozenDict

from fortuna.typing import (
    Mutable,
    Params,
)


class ModelManagerState:
    params: Params
    mutable: Optional[Mutable] = None

    def __init__(self, params: Params, mutable: Optional[Mutable] = None):
        """
        A model manager state class.

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
    def init_from_dict(cls, d: Union[Dict, FrozenDict]) -> ModelManagerState:
        """
        Initialize the model manager state from a dictionary. This dictionary should be like the output of
        :func:`~fortuna.model.model_manager.base.ModelManager.init`.

        Parameters
        ----------
        d : Union[Dict, FrozenDict]
            A dictionary like the output of :func:`~fortuna.model.model_manager.base.ModelManager.init`.

        Returns
        -------
        ModelManagerState
            An model manager state.
        """
        params = FrozenDict(
            {k: FrozenDict({"params": v["params"]}) for k, v in d.items()}
        )
        mutable = FrozenDict(
            {
                k: FrozenDict({_k: _v for _k, _v in v.items() if _k != "params"})
                for k, v in d.items()
            }
        )
        flag = 0
        for k, v in mutable.items():
            if len(v) > 0:
                flag += 1
        if flag == 0:
            mutable = None
        return cls(params=params, mutable=mutable)
