from typing import (
    Any,
    Callable,
    Dict,
    List,
    Tuple,
)

from jax._src.prng import PRNGKeyArray
from jax.experimental.pjit import pjit
import jax.numpy as jnp
from jax.sharding import PartitionSpec

from fortuna.data.loader import (
    DataLoader,
    TargetsLoader,
)
from fortuna.data.loader.base import ShardedPrefetchedLoader
from fortuna.output_calib_model.state import OutputCalibState
from fortuna.partitioner.partition_manager.base import PartitionManager
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
        state: OutputCalibState,
        batch: Batch,
        outputs: Array,
        loss_fun: Callable,
        rng: PRNGKeyArray,
        n_data: int,
    ) -> Tuple[OutputCalibState, Dict[str, Any]]:
        with self.partition_manager.partitioner.mesh:
            return pjit(
                super().training_step,
                static_argnums=(3, 5),
                in_shardings=(
                    self.partition_manager.shardings,
                    PartitionSpec(("dp", "fsdp")),
                    outputs.sharding,
                    PartitionSpec(),
                ),
            )(state, batch, outputs, loss_fun, rng, n_data)

    def validation_loss_step(
        self,
        state: OutputCalibState,
        batch: Batch,
        outputs: Array,
        loss_fun: Callable,
        rng: PRNGKeyArray,
        n_data: int,
    ) -> Dict[str, jnp.ndarray]:
        with self.partition_manager.partitioner.mesh:
            fun = super().validation_loss_step
            return pjit(
                fun,
                static_argnums=(3, 5),
                in_shardings=(
                    self.partition_manager.shardings,
                    PartitionSpec(("dp", "fsdp")),
                    outputs.sharding,
                    PartitionSpec(),
                ),
            )(state, batch, outputs, loss_fun, rng, n_data)

    def on_train_start(
        self,
        state: OutputCalibState,
        data_loaders: List[DataLoader],
        outputs_loaders: List[TargetsLoader],
        rng: PRNGKeyArray,
    ) -> Tuple[
        OutputCalibState,
        List[ShardedPrefetchedLoader],
        List[ShardedPrefetchedLoader],
        PRNGKeyArray,
    ]:
        state, data_loaders, output_loaders, rng = super(
            ShardingMixin, self
        ).on_train_start(state, data_loaders, outputs_loaders, rng)
        data_loaders = [
            ShardedPrefetchedLoader(
                loader=data_loader,
                partition_manager=self.partition_manager,
                partition_spec=PartitionSpec(("dp", "fsdp")),
            )
            for data_loader in data_loaders
        ]
        outputs_loaders = [
            ShardedPrefetchedLoader(
                loader=output_loader, partition_manager=self.partition_manager
            )
            for output_loader in output_loaders
        ]
        return state, data_loaders, outputs_loaders, rng
