from typing import (
    Callable,
    Dict,
    Optional,
    Tuple,
    Union,
)

from flax.core import FrozenDict
from jax import (
    jit,
    jvp,
    lax,
    vjp,
    vmap,
)
import jax.numpy as jnp
from jax.tree_util import tree_map

from fortuna.typing import (
    AnyKey,
    Array,
    InputData,
    Params,
)
from fortuna.utils.freeze import get_paths_with_label
from fortuna.utils.grad import value_and_jacobian_squared_row_norm
from fortuna.utils.nested_dicts import (
    nested_get,
    nested_set,
)


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


def sequential_probit_scaling(
    apply_fn: Callable[[Params, InputData], jnp.ndarray],
    params: Params,
    x: InputData,
    log_var: Union[float, Array],
    has_aux: bool = False,
    freeze_fun: Optional[Callable[[Tuple[AnyKey, ...], Array], str]] = None,
    top_k: Optional[int] = None,
    memory: Optional[int] = None,
    n_final_tokens: Optional[int] = None,
    stop_gradient: bool = False,
) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Dict]]:
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

    def _apply_fn(_p, _x, tau):
        _f = apply_fn(set_params(_p), _x)
        if has_aux:
            _f, _ = _f
        _f = _f[0]
        if _f.ndim > 1:
            _f = _f[tau]
        return _f

    f = apply_fn(params, x)
    if has_aux:
        f, aux = f

    if f.ndim > 3:
        raise ValueError("The model outputs can be at most three dimensional.")
    if f.ndim == 2:
        f = f[:, None]

    n_outputs = f.shape[-1]
    seq_length = f.shape[1]
    if n_final_tokens is None:
        n_final_tokens = seq_length
    if n_final_tokens <= 0 or n_final_tokens > seq_length:
        raise ValueError(
            f"`n_final_tokens` must be greater than 0 and cannot be greater than {seq_length}."
        )
    if memory is None:
        memory = n_final_tokens
    if memory <= 0 or memory > n_final_tokens:
        raise ValueError(
            f"`memory` must be greater than 0 and cannot be greater than {n_final_tokens}."
        )

    block_size = top_k if top_k is not None else n_outputs
    tot_size = memory * block_size
    batch_size = f.shape[0]

    indices = None
    if top_k is not None:
        indices = vmap(
            lambda _fx: vmap(lambda _fxtau: jnp.argsort(_fxtau)[-top_k:])(_fx)
        )(f)

    x = x[:, None] if not isinstance(x, dict) else tree_map(lambda v: v[:, None], x)

    def compute_cov(new_tau, prev_tau):
        new_tau -= 1
        prev_tau -= 1

        @vmap
        def _compute_cov(_x, idx):
            new_idx = idx[new_tau] if idx is not None else None
            prev_idx = idx[prev_tau] if idx is not None else None
            size = n_outputs if idx is None else len(prev_idx)

            new_fun = lambda p: (
                _apply_fn(p, _x, new_tau)[new_idx]
                if idx is not None
                else _apply_fn(p, _x, new_tau)
            )
            prev_fun = lambda p: (
                _apply_fn(p, _x, prev_tau)[prev_idx]
                if idx is not None
                else _apply_fn(p, _x, prev_tau)
            )

            J1J2T_op = lambda v: jvp(
                new_fun,
                (sub_params if params_paths is not None else params,),
                vjp(prev_fun, sub_params if params_paths is not None else params)[1](v),
            )[1]

            return vmap(J1J2T_op)(jnp.eye(size)).T

        return jnp.where(
            prev_tau != -1, _compute_cov(x, indices), jnp.empty(block_size)
        )

    init_tau = seq_length - n_final_tokens + 1

    @jit
    def compute_P(new_tau, old_taus):
        P = vmap(lambda tau: compute_cov(new_tau, tau), out_axes=2)(old_taus)
        return P.reshape(P.shape[0], P.shape[1], P.shape[2] * P.shape[3])

    @vmap
    def get_diag(mat):
        return jnp.diag(mat)

    def fun(carry, tau):
        Jinv, old_taus = carry
        S = compute_cov(tau, tau)

        P = compute_P(tau, old_taus)
        M = jnp.matmul(P, Jinv)
        C = S - jnp.matmul(M, P.swapaxes(1, 2))

        M = M[:, :, block_size:]
        Jinv = Jinv[:, block_size:, block_size:]
        Cinv = jnp.linalg.inv(C)
        MtCinv = jnp.matmul(M.swapaxes(1, 2), Cinv)

        Jinv = jnp.concatenate(
            (
                jnp.concatenate((Jinv + jnp.matmul(MtCinv, M), -MtCinv), axis=2),
                jnp.concatenate((-MtCinv.swapaxes(1, 2), Cinv), axis=2),
            ),
            axis=1,
        )

        old_taus = jnp.concatenate((old_taus[1:], jnp.array([tau])))
        return (Jinv, old_taus), get_diag(C)

    def get_diagCs(_params):
        old_taus = jnp.concatenate(
            (
                jnp.zeros(memory - 1, dtype="int32") - 1,
                jnp.array([init_tau], dtype="int32"),
            )
        )
        C = compute_cov(old_taus[-1], old_taus[-1])

        if n_final_tokens > 1:
            Jinv = jnp.linalg.inv(C)
            Jinv = jnp.concatenate(
                (
                    jnp.zeros((batch_size, (memory - 1) * block_size, tot_size)),
                    jnp.concatenate(
                        (
                            jnp.zeros(
                                (batch_size, block_size, (memory - 1) * block_size)
                            ),
                            Jinv,
                        ),
                        axis=2,
                    ),
                ),
                axis=1,
            )

            _, diagCs = lax.scan(
                fun, (Jinv, old_taus), jnp.arange(init_tau + 1, seq_length + 1)
            )
            diagCs = jnp.concatenate(
                (get_diag(C)[:, None], diagCs.swapaxes(0, 1)), axis=1
            )
        else:
            diagCs = get_diag(C)[:, None]

        return diagCs

    diagCs = get_diagCs(params if sub_params is None else sub_params)
    if stop_gradient:
        diagCs = lax.stop_gradient(diagCs)

    if top_k is not None:
        scales = jnp.max(diagCs, axis=2, keepdims=True).repeat(n_outputs, axis=2)
        scales = vmap(
            lambda i: vmap(
                lambda j: scales[i, j]
                .at[indices[i, j]]
                .set(diagCs[i, j, indices[i, j]])
            )(jnp.arange(seq_length - n_final_tokens, seq_length))
        )(jnp.arange(batch_size))
    else:
        scales = diagCs

    f = jnp.concatenate(
        (
            f[:, : seq_length - n_final_tokens],
            f[:, seq_length - n_final_tokens :]
            / (1 + jnp.pi / 8 * jnp.exp(log_var) * scales),
        ),
        axis=1,
    )

    if seq_length == 1:
        f = f[:, 0]

    if has_aux:
        return f, aux
    return f
