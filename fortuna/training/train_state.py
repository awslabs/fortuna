from __future__ import annotations

from typing import (
    Any,
    Optional,
)

from flax.training import (
    dynamic_scale,
    train_state,
)
import jax.numpy as jnp

from fortuna.typing import Params
from fortuna.utils.strings import convert_string_to_jnp_array


class TrainState(train_state.TrainState):
    encoded_name: jnp.ndarray = convert_string_to_jnp_array("TrainState")
    frozen_params: Optional[Params] = None
    dynamic_scale: Optional[dynamic_scale.DynamicScale] = None

    @classmethod
    def init(cls, *args, **kwargs) -> Any:
        pass

    @classmethod
    def init_from_dict(cls, *args, **kwargs) -> TrainState:
        pass
