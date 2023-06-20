from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Tuple,
    Union,
)

import flax.linen as nn
import jax.numpy as jnp

from fortuna.model_editor.base import ModelEditor
from fortuna.typing import (
    AnyKey,
    Array,
    InputData,
    Params,
)
from fortuna.utils.probit import sequential_probit_scaling


class ProbitClassificationModelEditor(ModelEditor):
    freeze_fun: Optional[Callable[[Tuple[AnyKey, ...], Array], str]] = None
    top_k: Optional[int] = None
    memory: Optional[int] = None
    n_final_tokens: Optional[int] = None

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
        outputs = sequential_probit_scaling(
            apply_fn,
            model_params,
            x,
            log_var=log_var,
            has_aux=has_aux,
            freeze_fun=self.freeze_fun,
            top_k=self.top_k,
            memory=self.memory,
            n_final_tokens=self.n_final_tokens
        )
        return outputs
