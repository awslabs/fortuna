import os
from copy import deepcopy
from typing import Dict, List, Optional, Union

from fortuna.training.mixin import WithCheckpointingMixin
from fortuna.training.train_state import TrainState
from fortuna.typing import OptaxOptimizer, Path


class TrainStateRepository(WithCheckpointingMixin):
    def __init__(self, checkpoint_dir: Optional[Path] = None):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.__state = None

    def get(
        self,
        checkpoint_path: Optional[Path] = None,
        optimizer: Optional[OptaxOptimizer] = None,
        prefix: str = "checkpoint_",
        **kwargs
    ) -> Union[Dict, TrainState]:
        if not checkpoint_path and not self.checkpoint_dir and not self.__state:
            raise ValueError("No state available.")
        if checkpoint_path or self.checkpoint_dir:
            return self.restore_checkpoint(
                restore_checkpoint_path=checkpoint_path or self.checkpoint_dir,
                optimizer=optimizer,
                prefix=prefix,
                **kwargs
            )
        if optimizer is not None:
            self.__state = self.__state.replace(tx=optimizer, opt_state=optimizer.init(self.__state.params))
            # return self.__state.init_from_dict(vars(self.__state), optimizer=optimizer)
        return deepcopy(self.__state)

    def put(
        self,
        state: TrainState,
        checkpoint_path: Optional[Path] = None,
        keep: int = 1,
        prefix: str = "checkpoint_",
    ) -> None:
        if checkpoint_path or self.checkpoint_dir:
            self.save_checkpoint(
                state=state,
                save_checkpoint_dir=checkpoint_path or self.checkpoint_dir,
                keep=keep,
                force_save=True,
                prefix=prefix,
            )
        else:
            self.__state = state

    def pull(
        self,
        checkpoint_path: Path = None,
        optimizer: Optional[OptaxOptimizer] = None,
        prefix: str = "checkpoint_",
        **kwargs
    ) -> TrainState:
        state = self.get(
            checkpoint_path=checkpoint_path,
            optimizer=optimizer,
            prefix=prefix,
            **kwargs
        )
        if checkpoint_path or self.checkpoint_dir:
            os.remove(
                checkpoint_path
                or self.get_path_latest_checkpoint(self.checkpoint_dir, prefix=prefix)
            )
        return state

    def update(
        self,
        variables: Dict,
        checkpoint_path: Path = None,
        optimizer: Optional[OptaxOptimizer] = None,
        keep: int = 1,
        prefix: str = "checkpoint_",
        **kwargs
    ):
        state = self.pull(
            checkpoint_path=checkpoint_path,
            optimizer=optimizer,
            prefix=prefix,
            **kwargs
        )
        state = state.replace(**variables)
        self.put(state, checkpoint_path=checkpoint_path, keep=keep, prefix=prefix)

    def extract(
        self,
        keys: List[str],
        checkpoint_path: Optional[Path] = None,
        prefix: str = "checkpoint_",
        **kwargs
    ) -> Dict:
        state = self.get(checkpoint_path=checkpoint_path, prefix=prefix, **kwargs)
        return {k: getattr(state, k) for k in keys}
