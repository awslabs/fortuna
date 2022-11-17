from __future__ import annotations

from typing import Any, Dict, Optional

import jax.numpy as jnp
from fortuna.prob_model.posterior.map.map_state import MAPState
from fortuna.prob_model.posterior.state import PosteriorState
from fortuna.typing import OptaxOptimizer
from fortuna.utils.strings import convert_string_to_jnp_array


class SWAGState(PosteriorState):
    """
    Attributes
    ----------
    encoded_name: jnp.ndarray
        SWAG state name encoded as an array.
    mean: Optional[jnp.ndarray]
        Mean of the posterior approximation.
    std: Optional[jnp.ndarray]
        Diagonal standard deviation of the posterior approximation.
    dev: Optional[jnp.ndarray]
        Deviation term of the covariance matrix of the posterior approximation.
    """

    mean: Optional[jnp.ndarray] = None
    std: Optional[jnp.ndarray] = None
    dev: Optional[jnp.ndarray] = None
    encoded_name: jnp.ndarray = convert_string_to_jnp_array("SWAGState")

    @classmethod
    def convert_from_map_state(
        cls, map_state: MAPState, optimizer: OptaxOptimizer
    ) -> SWAGState:
        """
        Convert a MAP state into a SWAG state.

        Parameters
        ----------
        map_state: MAPState
            A MAP posterior state.
        optimizer: OptaxOptimizer
            An Optax optimizer.

        Returns
        -------
        SWAGState
            A SWAG state.
        """
        return SWAGState.init(
            params=map_state.params,
            mutable=map_state.mutable,
            optimizer=optimizer,
            calib_params=map_state.calib_params,
            calib_mutable=map_state.calib_mutable,
        )

    def update(self, variables: Dict[str, Any]) -> SWAGState:
        """
        Update the SWAG state.

        Parameters
        ----------
        variables: Dict[str, Any]
            The attributes to update and their values.

        Returns
        -------
        SWAGState
            Updated SWAG state.
        """
        unchanged_keys = {k: v for k, v in vars(self).items() if k not in variables}
        return self.replace(**unchanged_keys, **variables)
