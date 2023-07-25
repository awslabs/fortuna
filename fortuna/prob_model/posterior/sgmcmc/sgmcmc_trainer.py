from fortuna.prob_model.posterior.map.map_trainer import MAPTrainer
from typing import Dict, List
import jax.numpy as jnp
from fortuna.training.train_state import TrainState
from fortuna.training.mixins.jitted import JittedMixin
from fortuna.training.mixins.multi_device import MultiDeviceMixin


class SGMCMCTrainer(MAPTrainer):
    def on_train_end(self, state: TrainState, mark_checkpoint_as_last: bool = False) -> TrainState:
        return super().on_train_end(state, mark_checkpoint_as_last)

    def validation_epoch_end(
        self,
        validation_losses_and_metrics_current_epoch: List[Dict[str, jnp.ndarray]],
        state: TrainState,
        mark_checkpoint_as_best: bool = False
    ) -> Dict[str, float]:
        return super().validation_epoch_end(validation_losses_and_metrics_current_epoch, state, mark_checkpoint_as_best)


class JittedSGMCMCTrainer(JittedMixin, SGMCMCTrainer):
    pass


class MultiDeviceSGMCMCTrainer(MultiDeviceMixin, SGMCMCTrainer):
    pass
