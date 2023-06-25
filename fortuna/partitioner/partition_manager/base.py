from typing import (
    Any,
    Callable,
    List,
    Optional,
)

from jax import (
    device_put,
    eval_shape,
    random,
)
from jax._src.prng import PRNGKeyArray
from jax.experimental.pjit import pjit
from jax.sharding import (
    NamedSharding,
    PartitionSpec,
)
from jax.tree_util import (
    tree_map,
    tree_map_with_path,
)

from fortuna.partitioner.base import Partitioner
from fortuna.training.train_state import TrainState
from fortuna.utils.partition import match_partition_specs
from fortuna.utils.random import WithRNG


class PartitionManager(WithRNG):
    def __init__(self, partitioner: Partitioner):
        self.partitioner = partitioner
        self._shapes_dtypes = None
        self._shardings = None

    @property
    def shapes_dtypes(self):
        return self._shapes_dtypes

    @shapes_dtypes.setter
    def shapes_dtypes(self, shapes_dtypes: TrainState):
        self._shapes_dtypes = shapes_dtypes
        partitions = match_partition_specs(self.partitioner.specs, self._shapes_dtypes)
        self._shardings = tree_map(
            lambda p: NamedSharding(mesh=self.partitioner.mesh, spec=p), partitions
        )

    @property
    def shardings(self):
        return self._shardings

    @shardings.setter
    def shardings(self, shardings: Optional[TrainState]):
        self._shardings = shardings

    def init_sharded_state(self, init_state_fn: Callable[[Any], TrainState], *args):
        self.shapes_dtypes = eval_shape(init_state_fn, random.PRNGKey(0))

        with self.partitioner.mesh:
            return pjit(
                init_state_fn,
                in_shardings=PartitionSpec(),
                out_shardings=self.shardings,
            )(*args)

    def reshard(
        self, state: TrainState, exclude: Optional[List[str]] = None
    ) -> TrainState:
        if self.shardings is not None:
            if exclude is None:
                exclude = []
            return tree_map_with_path(
                lambda p, _v, s: device_put(_v, s)
                if _v is not None and p[0].name not in exclude
                else _v,
                state,
                self.shardings,
            )
