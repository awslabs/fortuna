from typing import Dict

from jax import local_device_count
from jax.experimental.mesh_utils import create_device_mesh
from jax.interpreters import pxla
from jax.lax import with_sharding_constraint
from jax.sharding import (
    Mesh,
    PartitionSpec,
)
import numpy as np

from fortuna.utils.partition.base import get_names_from_partition_spec


def get_mesh(axis_dims: Dict[str, int]):
    keys = tuple(axis_dims.keys())
    dims = tuple(axis_dims.values())

    allowed_keys = ("dp", "fsdp", "mp")
    if set(keys) != set(allowed_keys):
        raise ValueError(
            f"`axis_dims` must contain exactly the following keys: {allowed_keys}."
        )
    for v in dims:
        if type(v) != int:
            raise ValueError("All values in `axes_dims` must be integers or `-1`.")
    if len(np.where(np.array(dims) == -1)[0]) > 1:
        raise ValueError("At most one axis dimension can be `-1`.")

    n_devices = local_device_count()

    fixed_prod = np.prod([v for v in dims if v != -1])
    reminder = n_devices % fixed_prod
    if fixed_prod > n_devices:
        raise ValueError(
            f"The product of the specified axes dimensions cannot be greater than {n_devices}, "
            f"the number of available devices."
        )
    if reminder != 0:
        raise ValueError(
            "The product of the axis dimensions must divide the number of available devices. "
            f"However, {n_devices} were found, and {fixed_prod} to be the product of the specified axis "
            f"dimensions."
        )

    dims = tuple([dims[np.where(np.array(keys) == k)[0][0]] for k in allowed_keys])
    mesh_shape = np.arange(n_devices).reshape(dims).shape
    physical_mesh = create_device_mesh(mesh_shape)
    return Mesh(physical_mesh, allowed_keys)


def names_in_current_mesh(*names) -> bool:
    """
    Check if the axis names in the current mesh contain the names provided.

    Parameters
    ----------
    names: List[str]
        Provided names.

    Returns
    -------
    bool:
        Whether the list of provided names is contained in the list of axis names from the current mesh.
    """
    mesh_axis_names = pxla.thread_resources.env.physical_mesh.axis_names
    return set(names) <= set(mesh_axis_names)


def with_conditional_sharding_constraint(x, partition_specs):
    """

    Parameters
    ----------
    x
    partition_specs

    Returns
    -------

    """
    """ A smarter version of with_sharding_constraint that only applies the
        constraint if the current mesh contains the axes in the partition specs.
    """
    axis_names = get_names_from_partition_spec(partition_specs)
    if names_in_current_mesh(*axis_names):
        x = with_sharding_constraint(x, partition_specs)
    return x
