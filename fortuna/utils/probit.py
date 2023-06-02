from typing import (
    Callable,
    Dict,
    Tuple,
    Union,
)

import jax.numpy as jnp

from fortuna.typing import (
    Array,
    InputData,
    Params,
)
from fortuna.utils.grad import value_and_jacobian_squared_row_norm


def probit_scaling(
    apply_fn: Callable[[Params, InputData], jnp.ndarray],
    params: Params,
    x: InputData,
    log_var: Union[float, Array],
    has_aux: bool = False,
) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Dict]]:
    f, jacobian_squared_row_norm = value_and_jacobian_squared_row_norm(
        apply_fn, params, x, has_aux=has_aux
    )
    if has_aux:
        f, aux = f
    f /= 1 + jnp.pi / 8 * jnp.exp(log_var) * jacobian_squared_row_norm

    if has_aux:
        return f, aux
    return f
