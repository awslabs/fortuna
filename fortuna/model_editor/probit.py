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


class ProbitModelEditor(ModelEditor):
    freeze_fun: Optional[Callable[[Tuple[AnyKey, ...], Array], str]] = None
    top_k: Optional[int] = None
    memory: Optional[int] = None
    n_final_tokens: Optional[int] = None
    init_log_var: float = -5.0
    stop_gradient: bool = False

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
        log_var = self.param(
            "log_var", nn.initializers.constant(self.init_log_var), (1,)
        )
        outputs = sequential_probit_scaling(
            apply_fn,
            model_params,
            x,
            log_var=log_var,
            has_aux=has_aux,
            freeze_fun=self.freeze_fun,
            top_k=self.top_k,
            memory=self.memory,
            n_final_tokens=self.n_final_tokens,
            stop_gradient=self.stop_gradient
        )
        return outputs
