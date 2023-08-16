from typing import (
    Callable,
    Dict,
    Optional,
    Tuple,
    Union,
)

from flax.core import FrozenDict
from jax import (
    jacrev,
    vmap,
)
import jax.numpy as jnp
from jax.tree_util import (
    tree_map,
    tree_reduce,
)

from fortuna.typing import (
    AnyKey,
    Array,
    InputData,
    Params,
)
from fortuna.utils.freeze import get_paths_with_label
from fortuna.utils.nested_dicts import (
    nested_get,
    nested_set,
)


def value_and_jacobian_squared_row_norm(
    apply_fn: Callable[
        [Params, InputData], Union[jnp.ndarray, Tuple[jnp.ndarray, Dict]]
    ],
    params: Params,
    x: InputData,
    has_aux: bool = False,
    freeze_fun: Optional[Callable[[Tuple[AnyKey, ...], Array], str]] = None,
    top_k: Optional[int] = None,
) -> Tuple[Union[jnp.ndarray, Tuple[jnp.ndarray, Dict]], jnp.ndarray]:
    params = params.unfreeze()

    params_paths = None
    sub_params = None
    if freeze_fun is not None:
        params_paths = tuple(
            get_paths_with_label(
                params, freeze_fun, label=True, allowed_labels=[True, False]
            )
        )
        sub_params = tuple([nested_get(d=params, keys=path) for path in params_paths])

    def set_params(_p):
        if params_paths is None:
            return _p
        return FrozenDict(nested_set(d=params, key_paths=params_paths, objs=_p))

    def _apply_fn(_p, _x):
        _f = apply_fn(set_params(_p), _x)
        if has_aux:
            _f, _ = _f
        return _f[0]

    f = apply_fn(params, x)
    if has_aux:
        f, aux = f

    n_dim = f.shape[-1]

    indices = None
    if top_k is not None:
        indices = vmap(lambda _f: jnp.argsort(_f)[-top_k:])(f)

    x = x[:, None] if not isinstance(x, dict) else tree_map(lambda v: v[:, None], x)

    @vmap
    def jacobian_squared_row_norm_fn(_x, idx):
        _row_norms = tree_reduce(
            lambda a, b: a + jnp.sum(b**2, axis=tuple(range(1, b.ndim))),
            jacrev(
                lambda p: _apply_fn(p, _x)[idx] if idx is not None else _apply_fn(p, _x)
            )(sub_params if params_paths is not None else params),
            initializer=0,
        )
        if idx is None:
            return _row_norms
        row_norms = jnp.max(_row_norms) * jnp.ones(n_dim)
        row_norms = row_norms.at[idx].set(_row_norms)
        return row_norms

    jac_squared_row_norm = jacobian_squared_row_norm_fn(x, indices)

    if has_aux:
        return (f, aux), jac_squared_row_norm
    return f, jac_squared_row_norm
