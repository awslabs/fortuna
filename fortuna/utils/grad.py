from functools import partial
from typing import (
    Callable,
    Dict,
    Tuple,
    Union,
)

from jax import (
    grad,
    lax,
    vjp,
    vmap,
)
import jax.numpy as jnp
from jax.tree_util import (
    tree_map,
    tree_reduce,
)

from fortuna.typing import (
    InputData,
    Params,
)


def value_and_jacobian_squared_row_norm(
    apply_fn: Callable[
        [Params, InputData], Union[jnp.ndarray, Tuple[jnp.ndarray, Dict]]
    ],
    params: Params,
    x: InputData,
    has_aux: bool = False,
) -> Tuple[Union[jnp.ndarray, Tuple[jnp.ndarray, Dict]], jnp.ndarray]:
    def _apply_fn(_p, _x):
        _f = apply_fn(_p, _x)
        if has_aux:
            _f, _ = _f
        return _f[0]

    f = apply_fn(params, x)
    if has_aux:
        f, aux = f
    n_dim = f.shape[-1]

    def jacobian_squared_row_norm_fn(i):
        return tree_reduce(
            lambda a, b: a + jnp.sum(b**2),
            vmap(lambda _x: grad(lambda p: _apply_fn(p, _x)[i])(params))(
                x[:, None]
                if not isinstance(x, dict)
                else tree_map(lambda v: v[:, None], x)
            ),
            initializer=0,
        )

    jac_squared_row_norm = jnp.array(
        lax.map(jacobian_squared_row_norm_fn, jnp.arange(n_dim))
    )

    if has_aux:
        return (f, aux), jac_squared_row_norm
    return f, jac_squared_row_norm
