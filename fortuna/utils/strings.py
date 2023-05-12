from typing import (
    Dict,
    List,
    Optional,
    Tuple,
)

import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_map

from fortuna.typing import Array


def convert_string_to_jnp_array(s: str) -> jnp.ndarray:
    return jnp.array([ord(c) for c in s])


def convert_string_to_np_array(s: str) -> np.ndarray:
    return np.array([ord(c) for c in s])


def encode_tuple_of_lists_of_strings_to_numpy(
    a: Optional[Tuple[List[str]]],
) -> Optional[Tuple[List[Array]]]:
    return (
        tuple([[convert_string_to_np_array(s) for s in key_path] for key_path in a])
        if a is not None
        else None
    )


def decode_encoded_tuple_of_lists_of_strings_to_array(
    encoded: Optional[Dict[str, List[Array]]]
) -> Optional[Tuple[List[str], ...]]:
    if encoded is None:
        return None
    encoded = tree_map(lambda v: "".join([chr(o) for o in v]), encoded)
    if isinstance(encoded, dict):
        return tuple([list(v.values()) for k, v in encoded.items()])
    else:
        return encoded
