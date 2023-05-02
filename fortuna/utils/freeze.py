from fortuna.typing import OptaxOptimizer, Array, Params, AnyKey
from flax.traverse_util import path_aware_map, flatten_dict
from flax.core.frozen_dict import freeze
from jax.tree_util import tree_leaves
from typing import Callable, Tuple, Any, Iterable, List, Optional
from optax import set_to_zero, multi_transform


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
            raise ValueError(f"All values must belong to one of the following labels: {labels}. However, "
                             f"value {v} was found.")


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
    return _get_paths_with_label(params, freeze_fun, "trainable")


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
    return _get_paths_with_label(params, freeze_fun, "frozen")


def _get_paths_with_label(
        params: Params,
        freeze_fun: Optional[Callable[[Tuple[AnyKey, ...], Array], str]],
        label: str
) -> Tuple[List[AnyKey], ...]:
    paths = path_aware_map(lambda p, v: p, params)
    if freeze_fun is not None:
        conds = list(flatten_dict(path_aware_map(freeze_fun, params)).values())
        all_values_in_labels(conds, ["trainable", "frozen"])
        return tuple([list(p) for c, p in zip(conds, flatten_dict(paths)) if c == label])
    else:
        return tuple([list(p) for p in paths])

