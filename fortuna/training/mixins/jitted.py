from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Tuple,
)

from flax import jax_utils
from flax.core import FrozenDict
import jax
from jax._src.prng import PRNGKeyArray
import jax.numpy as jnp
from optax._src.base import PyTree

from fortuna.training.train_state import TrainState
from fortuna.typing import (
    Array,
    Batch,
)


class JittedMixin:
    @partial(jax.jit, static_argnums=(0, 3, 5, 6, 7))
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

    @partial(jax.jit, static_argnums=(0, 3, 5, 6, 7, 8))
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
        return super().validation_step(
            state, batch, loss_fun, rng, n_data, metrics, unravel, kwargs
        )
