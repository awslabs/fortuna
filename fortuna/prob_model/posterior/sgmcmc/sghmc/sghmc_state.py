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
