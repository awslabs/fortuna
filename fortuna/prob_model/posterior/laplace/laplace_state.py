from __future__ import annotations

from typing import List, Tuple, Union, Optional, Dict

import jax.numpy as jnp
from flax.core import FrozenDict

from fortuna.prob_model.posterior.map.map_state import MAPState
from fortuna.prob_model.posterior.state import PosteriorState
from fortuna.typing import Params, AnyKey, Array
from fortuna.utils.nested_dicts import nested_pair
from fortuna.utils.strings import encode_tuple_of_lists_of_strings_to_numpy, convert_string_to_jnp_array


class LaplaceState(PosteriorState):
    """
    Attributes
    ----------
    encoded_name: jnp.ndarray
        Laplace state name encoded as an array.
    """

    encoded_name: jnp.ndarray = convert_string_to_jnp_array("LaplaceState")
    _encoded_which_params: Optional[Dict[str, Array]] = None

    @classmethod
    def convert_from_map_state(
        cls,
        map_state: MAPState,
        std: Union[Params, Tuple[Params, ...]],
        which_params: Tuple[List[AnyKey], ...],
    ) -> LaplaceState:
        """
        Convert a MAP state into a Laplace state.

        Parameters
        ----------
        map_state: MAPState
            A MAP state.
        std: Union[Params, Tuple[Params, ...]]
            Standard deviation parameters.
        which_params: Tuple[List[AnyKey], ...]
            Sequences of keys pointing to the parameters over which `std` is defined. If `which_params` is None,
            `std` must be defined for all parameters.

        Returns
        -------
        LaplaceState
            A Laplace state instance.
        """
        params = map_state.params.unfreeze()
        if which_params is not None:
            params = nested_pair(
                d=params,
                key_paths=which_params,
                objs=std,
                labels=("mean", "std"),
            )
        else:
            for k, v in params.items():
                params[k] = FrozenDict(
                    {"params": dict(mean=v["params"], std=std[k]["params"])}
                )
        d = vars(
            map_state.replace(
                params=FrozenDict(params), encoded_name=LaplaceState.encoded_name
            )
        )
        d["_encoded_which_params"] = encode_tuple_of_lists_of_strings_to_numpy(which_params)
        return LaplaceState.init_from_dict(d)
