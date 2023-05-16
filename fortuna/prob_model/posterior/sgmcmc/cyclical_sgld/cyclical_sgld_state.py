from __future__ import annotations

import jax.numpy as jnp

from fortuna.prob_model.posterior.state import PosteriorState
from fortuna.utils.strings import convert_string_to_jnp_array
from fortuna.prob_model.posterior.map.map_state import MAPState
from fortuna.typing import OptaxOptimizer


class CyclicalSGLDState(PosteriorState):
    """
    Attributes
    ----------
    encoded_name: jnp.ndarray
        CyclicalSGLDState state name encoded as an array.
    """

    encoded_name: jnp.ndarray = convert_string_to_jnp_array("CyclicalSGLDState")

    @classmethod
    def convert_from_map_state(
        cls, map_state: MAPState, optimizer: OptaxOptimizer
    ) -> CyclicalSGLDState:
        """
        Convert a MAP state into an CyclicalSGLDState state.

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
        return CyclicalSGLDState.init(
            params=map_state.params,
            mutable=map_state.mutable,
            optimizer=optimizer,
            calib_params=map_state.calib_params,
            calib_mutable=map_state.calib_mutable,
        )
