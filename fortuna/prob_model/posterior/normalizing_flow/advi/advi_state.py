from __future__ import annotations

from typing import (
    Dict,
    List,
    Optional,
)

import jax.numpy as jnp

from fortuna.prob_model.posterior.normalizing_flow.normalizing_flow_state import (
    NormalizingFlowState,
)
from fortuna.typing import Array
from fortuna.utils.strings import convert_string_to_jnp_array


class ADVIState(NormalizingFlowState):
    """
    Attributes
    ----------
    encoded_name: jnp.ndarray
        ADVI state name encoded as an array.
    """

    encoded_name: jnp.ndarray = convert_string_to_jnp_array("ADVIState")
    _encoded_which_params: Optional[Dict[str, List[Array]]] = None
