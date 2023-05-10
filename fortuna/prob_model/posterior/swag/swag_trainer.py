from typing import Any, Callable, Dict, Optional, Tuple, List

import jax.numpy as jnp
from flax.core import FrozenDict
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map

from fortuna.prob_model.posterior.map.map_trainer import MAPTrainer
from fortuna.prob_model.posterior.swag.swag_state import SWAGState
from fortuna.training.callback import Callback
from fortuna.training.trainer import JittedMixin, MultiDeviceMixin
from fortuna.typing import Array, Batch, Params
from fortuna.utils.nested_dicts import nested_get


class SWAGTrainer(MAPTrainer):
    _mean_rav_params = None
    _mean_squared_rav_params = None
    _deviation_rav_params = None
    _which_params = None

    def _update_state_with_stats(self, state: SWAGState) -> SWAGState:
        var = self._mean_squared_rav_params - self._mean_rav_params**2
        var = jnp.maximum(var, 0.0)
        return state.update(
            dict(
                mean=self._mean_rav_params
                if not self.multi_device
                else self._mean_rav_params[None],
                std=jnp.sqrt(var) if not self.multi_device else jnp.sqrt(var)[None],
                dev=self._deviation_rav_params
                if not self.multi_device
                else self._deviation_rav_params[None],
            )
        )

    def training_step_end(
        self,
        current_epoch: int,
        state: SWAGState,
        aux: Dict[str, Any],
        batch: Batch,
        metrics: Optional[Tuple[Callable[[jnp.ndarray, Array], float], ...]] = None,
        callbacks: Optional[List[Callback]] = None,
        kwargs: FrozenDict[str, Any] = FrozenDict(),
    ) -> Tuple[SWAGState, Dict[str, jnp.ndarray]]:
        if "rank" not in kwargs:
            raise AttributeError(
                """`rank` must be available in `kwargs` during training."""
            )
        rav_params = ravel_pytree(
            tree_map(lambda x: x[0], self._get_params_to_ravel(state.params))
            if self.multi_device else self._get_params_to_ravel(state.params)
        )[0]
        if self._mean_rav_params is None:
            self._mean_rav_params = rav_params
            self._mean_squared_rav_params = rav_params**2
            self._deviation_rav_params = jnp.zeros((len(rav_params), 1))
        else:
            self._mean_rav_params *= current_epoch
            self._mean_rav_params += rav_params
            self._mean_rav_params /= current_epoch + 1

            self._mean_squared_rav_params *= current_epoch
            self._mean_squared_rav_params += rav_params**2
            self._mean_squared_rav_params /= current_epoch + 1
            self._deviation_rav_params = jnp.concatenate(
                (
                    self._deviation_rav_params[:, -kwargs["rank"] + 1 :],
                    (rav_params - self._mean_rav_params)[:, None],
                ),
                axis=1,
            )
        if (self.save_checkpoint_dir is not None
            and self.save_every_n_steps is not None
            and self.save_every_n_steps > 0
            and self._global_training_step >= self.save_every_n_steps
            and self._global_training_step % self.save_every_n_steps == 0
        ):
            state = self._update_state_with_stats(state)
        return super().training_step_end(
            current_epoch, state, aux, batch, metrics, callbacks, kwargs
        )

    def _get_params_to_ravel(self, params: Params):
        if self._which_params is not None:
            return [nested_get(params, path) for path in self._which_params]
        return params


class SWAGJittedMixin(JittedMixin):
    def on_train_end(self, state: SWAGState) -> SWAGState:
        state = super().on_train_end(state)
        return self._update_state_with_stats(state)


class SWAGMultiDeviceMixin(MultiDeviceMixin):
    def on_train_end(self, state: SWAGState) -> SWAGState:
        state = self._update_state_with_stats(state)
        return super().on_train_end(state)


class JittedSWAGTrainer(SWAGJittedMixin, SWAGTrainer):
    pass


class MultiDeviceSWAGTrainer(SWAGMultiDeviceMixin, SWAGTrainer):
    pass
