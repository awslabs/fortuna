from __future__ import annotations

from typing import (
    Dict,
    List,
    Optional,
    Tuple,
)

import jax.numpy as jnp

from fortuna.prob_model.posterior.map.map_state import MAPState
from fortuna.prob_model.posterior.state import PosteriorState
from fortuna.typing import (
    AnyKey,
    Array,
    OptaxOptimizer,
)
from fortuna.utils.strings import (
    convert_string_to_jnp_array,
    encode_tuple_of_lists_of_strings_to_numpy,
)


class SGHMCState(PosteriorState):
    """
    Attributes
    ----------
    encoded_name: jnp.ndarray
        SGHMC state name encoded as an array.
    """

    encoded_name: jnp.ndarray = convert_string_to_jnp_array("SGHMCState")
    _encoded_which_params: Optional[Dict[str, List[Array]]] = None

    @classmethod
    def convert_from_map_state(
        cls,
        map_state: MAPState,
        optimizer: OptaxOptimizer,
        which_params: Tuple[List[AnyKey], ...],
    ) -> SGHMCState:
        """
        Convert a MAP state into an SGHMC state.

        Parameters
        ----------
        map_state: MAPState
            A MAP posterior state.
        optimizer: OptaxOptimizer
            An Optax optimizer.
        which_params: Tuple[List[AnyKey], ...]
            Sequences of keys pointing to the stochastic parameters.

        Returns
        -------
        SGHMCState
            An SGHMC state.
        """
        _encoded_which_params = encode_tuple_of_lists_of_strings_to_numpy(which_params)
        return cls.init(
            params=map_state.params,
            mutable=map_state.mutable,
            optimizer=optimizer,
            calib_params=map_state.calib_params,
            calib_mutable=map_state.calib_mutable,
            _encoded_which_params=_encoded_which_params,
        )
