from __future__ import annotations

import jax.numpy as jnp
from fortuna.prob_model.posterior.state import PosteriorState
from fortuna.typing import AnyKey
from fortuna.utils.strings import convert_string_to_jnp_array
from typing import Optional, Tuple, List


class ADVIState(PosteriorState):
    """
    Attributes
    ----------
    encoded_name: jnp.ndarray
        ADVI state name encoded as an array.
    """

    encoded_name: jnp.ndarray = convert_string_to_jnp_array("ADVIState")
    _encoded_which_params: Optional[Tuple[List[AnyKey], ...]] = None
