import re
from typing import (
    Dict,
    Tuple,
)

import jax.numpy as jnp
from jax.sharding import (
    PartitionSpec,
    Sharding,
)
from jax.tree_util import tree_map_with_path
import numpy as np
from optax._src.base import PyTree

from fortuna.utils.nested_dicts import path_to_string


def named_tree_map(f, tree, *rest, is_leaf=None, separator=None):
    return tree_map_with_path(
        lambda string_path, x, *r: f(
            path_to_string(string_path, separator=separator), x, *r
        ),
        tree,
        *rest,
        is_leaf=is_leaf,
    )


def match_partition_specs(
    partition_specs: Dict[str, PartitionSpec], tree: PyTree
) -> PyTree:
    """
    Match partition specifics to a tree structure.

    Parameters
    ----------
    partition_specs: Dict[str, Tuple[str]]
    tree: PyTree

    Returns
    -------
    PyTree
        A tree of partition specifics.
    """

    def get_partition_spec(path, shape_leaf):
        if len(shape_leaf.shape) == 0 or np.prod(shape_leaf.shape) == 1:
            # do not partition scalar values
            return PartitionSpec()
        for rule, ps in partition_specs.items():
            if re.search(rule, path) is not None:
                return ps
        # raise ValueError(f"A partition rule for the following path was not found: `{path}`")
        return PartitionSpec()

    return named_tree_map(get_partition_spec, tree, separator="/")


def get_names_from_partition_spec(partition_specs):
    """Return axis names from partition specs."""
    names = set()
    if isinstance(partition_specs, dict):
        partition_specs = partition_specs.values()
    for item in partition_specs:
        if item is None:
            continue
        elif isinstance(item, str):
            names.add(item)
        else:
            names.update(get_names_from_partition_spec(item))

    return list(names)
