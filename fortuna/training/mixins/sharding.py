from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)

from flax.core import FrozenDict
from jax import eval_shape
from jax._src.prng import PRNGKeyArray
from jax.experimental.pjit import pjit
import jax.numpy as jnp
from jax.sharding import PartitionSpec
from optax._src.base import PyTree

from fortuna.data.loader import DataLoader
from fortuna.data.loader.base import ShardedPrefetchedLoader
from fortuna.partitioner.partition_manager.base import PartitionManager
from fortuna.training.train_state import TrainState
from fortuna.typing import (
    Array,
    Batch,
)


class ShardingMixin:
    def __init__(self, *, partition_manager: PartitionManager, **kwargs):
        super().__init__(partition_manager=partition_manager, **kwargs)
        self.partition_manager = partition_manager

    def training_step(
        self,
        state: TrainState,
        batch: Batch,
        loss_fun: Callable,
        rng: PRNGKeyArray,
        n_data: int,
        unravel: Optional[Callable[[any], PyTree]] = None,
        kwargs: FrozenDict[str, Any] = FrozenDict(),
    ) -> Tuple[TrainState, Dict[str, Any]]:
        fun=super().training_step
        with self.partition_manager.partitioner.mesh:
            return pjit(
                fun,
                static_argnums=(2, 4, 5, 6),
                in_shardings=(
                    self.partition_manager.shardings,
                    PartitionSpec(("dp", "fsdp")),
                    PartitionSpec(),
                ),
                out_shardings=(
                    self.partition_manager.shardings,
                    PartitionSpec(),
                ),
            )(state, batch, loss_fun, rng, n_data, unravel, kwargs)

    def validation_step(
        self,
        state: TrainState,
        batch: Batch,
        loss_fun: Callable,
        rng: PRNGKeyArray,
        n_data: int,
        metrics: Optional[Tuple[Callable[[jnp.ndarray, Array], float], ...]] = None,
        unravel: Optional[Callable[[any], PyTree]] = None,
        kwargs: FrozenDict[str, Any] = FrozenDict(),
    ) -> Dict[str, jnp.ndarray]:
        with self.partition_manager.partitioner.mesh:
            return pjit(
                super().validation_step,
                static_argnums=(2, 4, 5, 6, 7),
                in_shardings=(
                    self.partition_manager.shardings,
                    PartitionSpec(("dp", "fsdp")),
                    PartitionSpec(),
                ),
            )(state, batch, loss_fun, rng, n_data, metrics, unravel, kwargs)

    def on_train_start(
        self, state: TrainState, data_loaders: List[DataLoader], rng: PRNGKeyArray
    ) -> Tuple[TrainState, List[ShardedPrefetchedLoader], PRNGKeyArray]:
        state, data_loaders, rng = super(ShardingMixin, self).on_train_start(
            state, data_loaders, rng
        )

        if self.freeze_fun is not None:
            self.partition_manager = PartitionManager(
                partitioner=self.partition_manager.partitioner
            )
            self.partition_manager.shapes_dtypes = eval_shape(lambda: state)

        data_loaders = [
            ShardedPrefetchedLoader(
                loader=dl,
                partition_manager=self.partition_manager,
                partition_spec=PartitionSpec(("dp", "fsdp")),
            )
            for dl in data_loaders
        ]
        return state, data_loaders, rng
