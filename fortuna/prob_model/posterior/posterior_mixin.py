from typing import Optional

from fortuna.prob_model.posterior.name_to_posterior_state import NameToPosteriorState
from fortuna.prob_model.posterior.state import PosteriorState
from fortuna.training.mixin import WithCheckpointingMixin
from fortuna.typing import (
    OptaxOptimizer,
    Path,
)


class WithPosteriorCheckpointingMixin(WithCheckpointingMixin):
    def restore_checkpoint(
        self,
        restore_checkpoint_path: Path,
        optimizer: Optional[OptaxOptimizer] = None,
        prefix: str = "checkpoint_",
        name_to_train_state: NameToPosteriorState = NameToPosteriorState,
        **kwargs,
    ) -> PosteriorState:
        return super().restore_checkpoint(
            restore_checkpoint_path,
            optimizer,
            prefix,
            name_to_train_state=name_to_train_state,
        )
