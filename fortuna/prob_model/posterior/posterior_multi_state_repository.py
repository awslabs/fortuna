import os
from typing import (
    Dict,
    List,
    Optional,
    Union,
)

from fortuna.prob_model.posterior.posterior_state_repository import (
    PosteriorStateRepository,
)
from fortuna.prob_model.posterior.state import PosteriorState
from fortuna.typing import (
    OptaxOptimizer,
    Path,
)
from fortuna.partitioner.partition_manager.base import PartitionManager
from fortuna.utils.checkpoint import get_checkpoint_manager
import pathlib
from orbax.checkpoint import CheckpointManager


class PosteriorMultiStateRepository:
    def __init__(
            self,
            size: int,
            partition_manager: Optional[PartitionManager] = None,
            checkpoint_manager: Optional[CheckpointManager] = None,
            checkpoint_type: Optional[str] = "last"
    ):
        self.size = size
        self.state = []
        for i in range(size):
            if checkpoint_manager is not None:
                path = pathlib.Path(checkpoint_manager.directory) / str(i)
                if checkpoint_type is not None:
                    path = path / checkpoint_type
                path = str(path)
            else:
                path = None
            self.state.append(
                PosteriorStateRepository(
                    partition_manager=partition_manager,
                    checkpoint_manager=get_checkpoint_manager(
                        checkpoint_dir=path,
                        keep_top_n_checkpoints=checkpoint_manager._options.max_to_keep if checkpoint_manager is not None else None
                    )
                )
            )

    def get(
        self,
        i: int = None,
        checkpoint_dir: Optional[Path] = None,
        optimizer: Optional[OptaxOptimizer] = None,
        **kwargs,
    ) -> Union[List[PosteriorState], PosteriorState]:
        def _get(_i):
            return self.state[_i].get(
                checkpoint_dir=checkpoint_dir,
                optimizer=optimizer,
                **kwargs,
            )

        if i is not None:
            return _get(i)
        state = []
        for i in range(self.size):
            state.append(_get(i))
        return state

    def put(
        self,
        state: PosteriorState,
        i: int = None,
        checkpoint_dir: Optional[Path] = None,
        keep: int = 1,
    ) -> None:
        def _put(_i):
            return self.state[_i].put(
                state=state, checkpoint_dir=checkpoint_dir, keep=keep
            )

        if i is not None:
            _put(i)
        else:
            for i in range(self.size):
                state.append(_put(i))

    def pull(
        self,
        i: int = None,
        checkpoint_dir: Path = None,
        optimizer: Optional[OptaxOptimizer] = None,
    ) -> PosteriorState:
        def _pull(_i):
            return self.state[_i].pull(
                checkpoint_dir=checkpoint_dir,
                optimizer=optimizer,
            )

        if i is not None:
            return _pull(i)
        state = []
        for i in range(self.size):
            state.append(_pull(i))
        return state

    def update(
        self,
        variables: Dict,
        i: int = None,
        checkpoint_dir: Path = None,
        optimizer: Optional[OptaxOptimizer] = None,
        keep: int = 1,
    ):
        def _update(_i):
            self.state[_i].update(
                variables=variables,
                checkpoint_dir=checkpoint_dir,
                optimizer=optimizer,
                keep=keep,
            )

        if i is not None:
            _update(i)
        else:
            for i in range(self.size):
                _update(i)

    def extract(
        self,
        keys: List[str],
        i: int = None,
        checkpoint_dir: Optional[Path] = None,
    ) -> Union[Dict, List[Dict]]:
        def _extract(_i):
            return self.state[_i].extract(
                keys=keys, checkpoint_dir=checkpoint_dir
            )

        if i is not None:
            return _extract(i)
        dicts = []
        for i in range(self.size):
            dicts.append(_extract(i))
        return dicts

    def extract_calib_keys(
        self,
        checkpoint_dir: Optional[Path] = None,
    ) -> Dict:
        return self.extract(
            ["calib_params", "calib_mutable"], 0, checkpoint_dir
        )
