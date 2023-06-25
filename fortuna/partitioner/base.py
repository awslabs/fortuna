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
        axis_dims: Optional[Dict[str, int]] = None,
        rules: Optional[Dict[str, Tuple[str, ...]]] = None,
        coordinator_address: Optional[str] = None,
        n_devices: Optional[int] = None,
    ):
        if axis_dims is None:
            axis_dims = {"dp": 1, "fsdp": 1, "mp": -1}
        if rules is None:
            rules = {}
        self.specs = {
            k: PartitionSpec(*v) if v is not None else PartitionSpec(None)
            for k, v in rules.items()
        }
        self.mesh = get_mesh(axis_dims)

        if coordinator_address is None:
            port = 8888
            while is_port_in_use(port):
                port += 1
            self.coordinator_address = f"localhost/{port}"
        else:
            self.coordinator_address = coordinator_address

        self.n_devices = n_devices
