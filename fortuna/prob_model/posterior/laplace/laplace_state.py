from __future__ import annotations

from typing import List, Tuple, Union

import jax.numpy as jnp
from flax.core import FrozenDict
from fortuna.prob_model.posterior.map.map_state import MAPState
from fortuna.prob_model.posterior.state import PosteriorState
from fortuna.typing import Params
from fortuna.utils.nested_dicts import nested_pair
from fortuna.utils.strings import convert_string_to_jnp_array


class LaplaceState(PosteriorState):
    """
    Attributes
    ----------
    encoded_name: jnp.ndarray
        Laplace state name encoded as an array.
    """

    encoded_name: jnp.ndarray = convert_string_to_jnp_array("LaplaceState")

    @classmethod
    def convert_from_map_state(
        cls,
        map_state: MAPState,
        std: Union[Params, Tuple[Params, ...]],
        which_params: Tuple[List],
    ) -> LaplaceState:
        """
        Convert a MAP state into a Laplace state.

        Parameters
        ----------
        map_state: MAPState
            A MAP state.
        std: Union[Params, Tuple[Params, ...]]
            Standard deviation parameters.
        which_params: Tuple[List]
            Sequences of keys pointing to the parameters over which `std` is defined. If `which_params` is None,
            `std` must be defined for all parameters.

        Returns
        -------
        LaplaceState
            A Laplace state instance.
        """
        params = map_state.params.unfreeze()
        if which_params:
            params = FrozenDict(
                nested_pair(params, which_params, std, ("mean", "std"),)
            )
        else:
            for k, v in params.items():
                params[k] = FrozenDict(
                    {"params": dict(mean=v["params"], std=std[k]["params"])}
                )
        map_state = map_state.replace(
            params=FrozenDict(params), encoded_name=LaplaceState.encoded_name
        )
        return LaplaceState.init_from_dict(vars(map_state))
