from typing import (
    Any,
    Callable,
    Dict,
    Tuple,
    Union,
)

import flax.linen as nn
import jax.numpy as jnp

from fortuna.model_editor.base import ModelEditor
from fortuna.typing import (
    InputData,
    Params,
)
from fortuna.utils.probit import probit_scaling


class ProbitClassificationModelEditor(ModelEditor):
    @nn.compact
    def __call__(
        self,
        apply_fn: Callable[
            [Params, InputData], Union[jnp.ndarray, Tuple[jnp.ndarray, Dict]]
        ],
        model_params: Params,
        x: Any,
        has_aux: bool,
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Dict]]:
        log_var = self.param("log_var", nn.initializers.zeros, (1,))
        return probit_scaling(
            apply_fn, model_params, x, log_var=log_var, has_aux=has_aux
        )
