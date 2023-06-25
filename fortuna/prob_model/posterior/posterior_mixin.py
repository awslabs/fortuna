from typing import Optional

from fortuna.prob_model.posterior.name_to_posterior_state import NameToPosteriorState
from fortuna.training.mixins.checkpointing import WithCheckpointingMixin
from fortuna.training.name_to_train_state import NameToTrainState
from fortuna.typing import (
    OptaxOptimizer,
    Path,
)


class WithPosteriorCheckpointingMixin(WithCheckpointingMixin):
    def restore_checkpoint(
        self,
        restore_checkpoint_dir: Path,
        optimizer: Optional[OptaxOptimizer] = None,
        name_to_train_state: NameToTrainState = NameToPosteriorState,
    ):
        return super().restore_checkpoint(
            restore_checkpoint_dir=restore_checkpoint_dir,
            optimizer=optimizer,
            name_to_train_state=name_to_train_state,
        )

    def get_shapes_dtypes_checkpoint(
        self,
        restore_checkpoint_dir: Optional[Path] = None,
        name_to_train_state: NameToTrainState = NameToPosteriorState,
    ):
        return super().get_shapes_dtypes_checkpoint(
            restore_checkpoint_dir=restore_checkpoint_dir,
            name_to_train_state=name_to_train_state,
        )
