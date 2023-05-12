from __future__ import annotations

from typing import Any

from flax.training import train_state
import jax.numpy as jnp

from fortuna.utils.strings import convert_string_to_jnp_array


class TrainState(train_state.TrainState):
    encoded_name: jnp.ndarray = convert_string_to_jnp_array("TrainState")

    @classmethod
    def init(cls, *args, **kwargs) -> Any:
        pass

    @classmethod
    def init_from_dict(cls, *args, **kwargs) -> TrainState:
        pass
