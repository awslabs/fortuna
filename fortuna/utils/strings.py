import jax.numpy as jnp


def convert_string_to_jnp_array(s: str) -> jnp.ndarray:
    return jnp.array([ord(c) for c in s])
