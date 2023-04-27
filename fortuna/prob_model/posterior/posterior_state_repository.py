from typing import Dict, Optional

from fortuna.prob_model.posterior.posterior_mixin import \
    WithPosteriorCheckpointingMixin
from fortuna.training.train_state_repository import TrainStateRepository
from fortuna.typing import Path


class PosteriorStateRepository(WithPosteriorCheckpointingMixin, TrainStateRepository):
    def extract_calib_keys(
        self,
        checkpoint_dir: Optional[Path] = None,
        prefix: str = "checkpoint_",
        **kwargs
    ) -> Dict:
        return super().extract(
            ["calib_params", "calib_mutable"], checkpoint_dir, prefix, **kwargs
        )
