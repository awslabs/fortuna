from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Tuple,
)

from flax.core import FrozenDict
from jax._src.prng import PRNGKeyArray
import jax.numpy as jnp
from optax._src.base import PyTree

from fortuna.prob_model.posterior.map.map_trainer import MAPTrainer
from fortuna.prob_model.posterior.sgmcmc.hmc import HMC_NAME
from fortuna.prob_model.posterior.sgmcmc.hmc.hmc_state import HMCState
from fortuna.training.trainer import (
    JittedMixin,
    MultiDeviceMixin,
)
from fortuna.typing import (
    Array,
    Batch,
)
from fortuna.utils.freeze import (
    has_multiple_opt_state,
    get_trainable_opt_state,
    update_trainable_opt_state,
)


class HMCTrainer(MAPTrainer):
    def training_step(
        self,
        state: HMCState,
        batch: Batch,
        loss_fun: Callable,
        rng: PRNGKeyArray,
        n_data: int,
        unravel: Optional[Callable[[any], PyTree]] = None,
        kwargs: FrozenDict[str, Any] = FrozenDict(),
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        state, aux = super().training_step(
            state=state,
            batch=batch,
            loss_fun=loss_fun,
            rng=rng,
            n_data=n_data,
            unravel=unravel,
            **kwargs,
        )
        if has_multiple_opt_state(state):
            opt_state = get_trainable_opt_state(state)._replace(log_prob=aux["loss"])
            state = update_trainable_opt_state(state, opt_state)
        else:
            opt_state = state.opt_state._replace(log_prob=aux["loss"])
            state = state.replace(opt_state=opt_state)
        return state, aux

    def __str__(self):
        return HMC_NAME


class JittedHMCTrainer(JittedMixin, HMCTrainer):
    pass


class MultiDeviceHMCTrainer(MultiDeviceMixin, HMCTrainer):
    pass
