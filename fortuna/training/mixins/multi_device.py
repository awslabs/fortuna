from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)

from flax import jax_utils
from flax.core import FrozenDict
import jax
from jax import (
    lax,
    random,
)
from jax._src.prng import PRNGKeyArray
import jax.numpy as jnp
from jax.tree_util import tree_map
from optax._src.base import PyTree

from fortuna.data.loader import DataLoader
from fortuna.training.callback import Callback
from fortuna.training.train_state import TrainState
from fortuna.typing import (
    Array,
    Batch,
)


class MultiDeviceMixin:
    all_reduce_mean = jax.pmap(lambda x: lax.pmean(x, "x"), "x")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multi_device = True

    @staticmethod
    def _add_device_dim_to_input_data_loader(data_loader: DataLoader) -> DataLoader:
        def _reshape_input_batch(batch):
            n_devices = jax.local_device_count()
            if batch.shape[0] % n_devices != 0:
                raise ValueError(
                    f"The size of all batches must be a multiple of {n_devices}, that is the number of "
                    f"available devices. Please set an appropriate batch size in the data loader."
                )
            single_input_shape = batch.shape[1:]
            # reshape to (local_devices, device_batch_size, *single_input_shape)
            return batch.reshape((n_devices, -1) + single_input_shape)

        class DataLoaderWrapper:
            def __init__(self, data_loader):
                self.data_loader = data_loader

            def __iter__(self):
                data_loader = map(
                    lambda batch: tree_map(_reshape_input_batch, batch),
                    self.data_loader,
                )
                data_loader = jax_utils.prefetch_to_device(data_loader, 2)
                yield from data_loader

        return (
            DataLoaderWrapper(data_loader) if data_loader is not None else data_loader
        )

    @staticmethod
    def _sync_mutable(state: TrainState) -> TrainState:
        return (
            state.replace(mutable=MultiDeviceMixin.all_reduce_mean(state.mutable))
            if state.mutable is not None
            else state
        )

    @staticmethod
    def _sync_array(arr: jnp.ndarray) -> jnp.ndarray:
        arr = lax.pmean(arr, axis_name="batch")
        return arr

    def _sync_state(self, state: TrainState) -> TrainState:
        state = self._sync_mutable(state)
        return jax.device_get(tree_map(lambda x: x[0], state))

    def on_train_start(
        self, state: TrainState, data_loaders: List[DataLoader], rng: PRNGKeyArray
    ) -> Tuple[TrainState, List[DataLoader], PRNGKeyArray]:
        state, data_loaders, rng = super(MultiDeviceMixin, self).on_train_start(
            state, data_loaders, rng
        )
        state = jax_utils.replicate(state)
        data_loaders = [
            self._add_device_dim_to_input_data_loader(dl) for dl in data_loaders
        ]
        model_key = random.split(rng, jax.local_device_count())
        return state, data_loaders, model_key

    def on_train_end(self, state: TrainState) -> TrainState:
        state = super(MultiDeviceMixin, self).on_train_end(state)
        return jax.device_get(tree_map(lambda x: x[0], state))

    def training_step_start(self, rng: PRNGKeyArray, step: int) -> PRNGKeyArray:
        step = step if isinstance(step, int) or step.ndim == 0 else step[0]
        return jax.vmap(lambda r: random.fold_in(r, step))(rng)

    @partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(0, 3, 5, 6, 7))
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
        return super().training_step(
            state, batch, loss_fun, rng, n_data, unravel, kwargs
        )

    def training_step_end(
        self,
        current_epoch: int,
        state: TrainState,
        aux: Dict[str, Any],
        batch: Batch,
        metrics: Optional[Tuple[Callable[[jnp.ndarray, Array], float], ...]],
        callbacks: Optional[List[Callback]] = None,
        kwargs: FrozenDict[str, Any] = FrozenDict(),
    ) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
        state, training_losses_and_metrics = super(
            MultiDeviceMixin, self
        ).training_step_end(
            current_epoch, state, aux, batch, metrics, callbacks, kwargs
        )
        return state, tree_map(lambda x: x.mean(), training_losses_and_metrics)

    def on_validation_start(self, state: TrainState) -> TrainState:
        state = super(MultiDeviceMixin, self).on_validation_start(state)
        state = self._sync_mutable(state)
        return state

    @partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(0, 3, 5, 6, 7, 8))
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
        validation_losses_and_metrics = super().validation_step(
            state, batch, loss_fun, rng, n_data, metrics, unravel, kwargs
        )
        return lax.pmean(validation_losses_and_metrics, axis_name="batch")
