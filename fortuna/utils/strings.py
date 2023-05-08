import jax.numpy as jnp
import numpy as np


def convert_string_to_jnp_array(s: str) -> jnp.ndarray:
    return jnp.array([ord(c) for c in s])


def convert_string_to_np_array(s: str) -> jnp.ndarray:
    return np.array([ord(c) for c in s])
