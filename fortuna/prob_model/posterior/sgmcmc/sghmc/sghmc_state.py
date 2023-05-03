from __future__ import annotations

import jax.numpy as jnp

from fortuna.prob_model.posterior.state import PosteriorState
from fortuna.utils.strings import convert_string_to_jnp_array
from fortuna.typing import Params


class SGHMCState(PosteriorState):
    """
    Attributes
    ----------
    encoded_name: jnp.ndarray
        SGHMC state name encoded as an array.
    """

    encoded_name: jnp.ndarray = convert_string_to_jnp_array("SGHMCState")

    @classmethod
    def convert_from_map_state(
        cls, map_state: MAPState, optimizer: OptaxOptimizer
    ) -> SGHMCState:
        """
        Convert a MAP state into an SGHMC state.

        Parameters
        ----------
        map_state: MAPState
            A MAP posterior state.
        optimizer: OptaxOptimizer
            An Optax optimizer.

        Returns
        -------
        SGHMCState
            An SGHMC state.
        """
        return SGHMCState.init(
            params=map_state.params,
            mutable=map_state.mutable,
            optimizer=optimizer,
            calib_params=map_state.calib_params,
            calib_mutable=map_state.calib_mutable,
        )
