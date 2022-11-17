from __future__ import annotations

import jax.numpy as jnp
from flax.core import FrozenDict
from fortuna.prob_model.posterior.map.map_state import MAPState
from fortuna.prob_model.posterior.state import PosteriorState
from fortuna.typing import Params
from fortuna.utils.strings import convert_string_to_jnp_array


class ADVIState(PosteriorState):
    """
    Attributes
    ----------
    encoded_name: jnp.ndarray
        ADVI state name encoded as an array.
    """

    encoded_name: jnp.ndarray = convert_string_to_jnp_array("ADVIState")

    @classmethod
    def convert_from_map_state(cls, map_state: MAPState, std: Params) -> ADVIState:
        """
        Convert a MAP state into an ADVI state.

        Parameters
        ----------
        map_state: MAPState
            A MAP posterior state.
        std: Params
            Standard deviation parameters.

        Returns
        -------
        ADVIState
            An ADVI state.
        """
        params = map_state.params.unfreeze()
        for k, v in std.items():
            params[k] = FrozenDict(
                {"params": dict(mean=params[k]["params"], logvar=v["params"])}
            )
        params = FrozenDict(params)
        return cls.init(
            params,
            map_state.mutable,
            map_state.tx,
            map_state.calib_params,
            map_state.calib_mutable,
        )
