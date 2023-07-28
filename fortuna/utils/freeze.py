from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

from flax.core.frozen_dict import freeze
from flax.traverse_util import (
    flatten_dict,
    path_aware_map,
)
from jax.tree_util import tree_leaves
from optax import (
    multi_transform,
    set_to_zero,
    MultiTransformState,
)
from optax._src.wrappers import MaskedState

from fortuna.typing import (
    AnyKey,
    Array,
    OptaxOptimizer,
    Params,
)

from fortuna.prob_model.posterior.state import PosteriorState


def all_values_in_labels(values: Iterable, labels: Any) -> None:
    """
    Check that all values belong to the given labels.

    Parameters
    ----------
    values: Iterable
        Some iterable of values.
    labels: Any
        A collection of labels
    """
    for v in values:
        if v not in labels:
            raise ValueError(
                f"All values must belong to one of the following labels: {labels}. However, "
                f"value {v} was found."
            )


def freeze_optimizer(
    params: Params,
    optimizer: OptaxOptimizer,
    freeze_fun: Optional[Callable[[Tuple[AnyKey, ...], Array], str]],
) -> OptaxOptimizer:
    """
    Given an optimizer, set its gradient update to zero in correspondence to the frozen parameters specified by the
    freeze function.

    Parameters
    ----------
    params: Params
        Model parameters.
    optimizer: OptaxOptimizer
        An optimizer to freeze.
    freeze_fun: Optional[Callable[[Tuple[AnyKey, ...], Array], str]]
        A freeze function. It takes in input a path and its correspondent array of parameters, and it returns
        "trainable" or "frozen". If "frozen", the gradient of those parameters will be set to zero. If the freeze
        function is None, the optimizer is returned as is.

    Returns
    -------
    OptaxOptimizer
        The same optimizer but frozen where specified by the freeze function.
    """
    if freeze_fun is None:
        return optimizer
    partition_params = path_aware_map(freeze_fun, params)
    all_values_in_labels(tree_leaves(partition_params), ["trainable", "frozen"])
    partition_params = freeze(partition_params)
    partition_optimizers = {"trainable": optimizer, "frozen": set_to_zero()}
    return multi_transform(partition_optimizers, partition_params)


def has_multiple_opt_state(state: PosteriorState):
    """
    Check if a given posterior state containts multiple optimizer states.

    Parameters
    ----------
    state: PosteriorState
        An instance of `PosteriorState`.

    Returns
    -------
    bool
    """
    return isinstance(state.opt_state, MultiTransformState)


def get_trainable_opt_state(state: PosteriorState):
    """
    Get a trainable optimizer state.

    Parameters
    ----------
    state: PosteriorState
        An instance of `PosteriorState`.

    Returns
    -------
    opt_state: Any
        An instance of trainable optimizer state.
    """
    return state.opt_state.inner_states["trainable"].inner_state


def update_trainable_opt_state(state: PosteriorState, opt_state: Any):
    """
    Update a trainable optimizer state.

    Parameters
    ----------
    state: PosteriorState
        An instance of `PosteriorState`.
    opt_state: Any
        An instance of trainable optimizer state.

    Returns
    -------
    PosteriorState
        An updated posterior state.
    """
    trainable_state = MaskedState(inner_state=opt_state)
    opt_state = MultiTransformState(
        inner_states={
            k: (trainable_state if k == "trainable" else v)
            for k, v in state.opt_state.inner_states.items()
        }
    )
    return state.replace(opt_state=opt_state)


def get_trainable_paths(
    params: Params,
    freeze_fun: Optional[Callable[[Tuple[AnyKey, ...], Array], str]],
) -> Tuple[List[AnyKey], ...]:
    """
    Return a tuple of sequences of keys pointing to the trainable parameters only.

    Parameters
    ----------
    params
    freeze_fun: Callable[[Tuple[AnyKey, ...], Array], str]
        A freeze function. It takes in input a path and its correspondent array of parameters, and it returns
        "trainable" or "frozen". If "frozen", the path corresponding to those parameters will not be returned.
        If the function is `None`, all parameters are marked as trainable.

    Returns
    -------
    Tuple[List[AnyKey], ...]
        A tuple of sequences of keys pointing to the trainable parameters only.
    """
    return get_paths_with_label(
        params, freeze_fun, "trainable", allowed_labels=["frozen", "trainable"]
    )


def get_frozen_paths(
    params: Params,
    freeze_fun: Optional[Callable[[Tuple[AnyKey, ...], Array], str]],
) -> Tuple[List[AnyKey], ...]:
    """
    Return a tuple of sequences of keys pointing to the frozen parameters only.

    Parameters
    ----------
    params
    freeze_fun: Callable[[Tuple[AnyKey, ...], Array], str]
        A freeze function. It takes in input a path and its correspondent array of parameters, and it returns
        "trainable" or "frozen". If "frozen", the path corresponding to those parameters will not be returned.
        If the function is `None`, all parameters are marked as trainable.

    Returns
    -------
    Tuple[List[AnyKey], ...]
        A tuple of sequences of keys pointing to the frozen parameters only.
    """
    return get_paths_with_label(
        params, freeze_fun, "frozen", allowed_labels=["frozen", "trainable"]
    )


def get_paths_with_label(
    params: Union[Params, Dict],
    freeze_fun: Optional[Callable[[Tuple[AnyKey, ...], Array], str]],
    label: Union[str, bool],
    allowed_labels: Optional[List[Union[str, bool]]] = None,
) -> Tuple[List[AnyKey], ...]:
    paths = path_aware_map(lambda p, v: p, params)
    if freeze_fun is not None:
        conds = list(flatten_dict(path_aware_map(freeze_fun, params)).values())
        all_values_in_labels(conds, allowed_labels)
        return tuple(
            [list(p) for c, p in zip(conds, flatten_dict(paths)) if c == label]
        )
    else:
        return tuple([list(p) for p in paths])
