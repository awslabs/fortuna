from typing import (
    Dict,
    Optional,
    Tuple,
)

from jax.sharding import PartitionSpec

from fortuna.utils.mesh import get_mesh
from fortuna.utils.port import is_port_in_use


class Partitioner:
    def __init__(
        self,
        axes_dims: Optional[Dict[str, int]] = None,
        rules: Optional[Dict[str, Tuple[str, ...]]] = None,
    ):
        if axes_dims is None:
            axes_dims = {"dp": 1, "fsdp": 1, "mp": -1}
        if rules is None:
            rules = {}
        self.specs = {
            k: PartitionSpec(*v) if v is not None else PartitionSpec(None)
            for k, v in rules.items()
        }
        self.mesh = get_mesh(axes_dims)
