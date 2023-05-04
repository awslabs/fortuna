import jax.numpy as jnp

from fortuna.prob_model.posterior.state import PosteriorState
from fortuna.utils.strings import convert_string_to_jnp_array


class CyclicalSGLDState(PosteriorState):
    """
    Attributes
    ----------
    encoded_name: jnp.ndarray
        CyclicalSGLDState state name encoded as an array.
    """

    encoded_name: jnp.ndarray = convert_string_to_jnp_array(
        "CyclicalSGLDState"
    )
