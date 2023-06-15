from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Tuple,
    Union,
)

import jax.numpy as jnp

from fortuna.typing import (
    AnyKey,
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
    freeze_fun: Optional[Callable[[Tuple[AnyKey, ...], Array], str]] = None,
    top_k: Optional[int] = None,
) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Dict]]:
    f, scale = value_and_jacobian_squared_row_norm(
        apply_fn, params, x, has_aux=has_aux, freeze_fun=freeze_fun, top_k=top_k
    )

    if has_aux:
        f, aux = f

    f /= 1 + jnp.pi / 8 * jnp.exp(log_var) * scale

    if has_aux:
        return f, aux
    return f
