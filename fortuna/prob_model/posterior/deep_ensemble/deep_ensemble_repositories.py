import os
from typing import Dict, List, Optional, Union

from fortuna.prob_model.posterior.deep_ensemble.deep_ensemble_state import \
    DeepEnsembleState
from fortuna.prob_model.posterior.posterior_state_repository import \
    PosteriorStateRepository
from fortuna.prob_model.posterior.state import PosteriorState
from fortuna.typing import OptaxOptimizer, Path


class DeepEnsemblePosteriorStateRepository:
    def __init__(self, ensemble_size: int, checkpoint_dir: Optional[Path] = None):
        self.ensemble_size = ensemble_size
        self.state = [
            PosteriorStateRepository(
                checkpoint_dir=os.path.join(checkpoint_dir, str(i))
                if checkpoint_dir
                else None
            )
            for i in range(ensemble_size)
        ]

    def get(
        self,
        i: int = None,
        checkpoint_path: Optional[Path] = None,
        optimizer: Optional[OptaxOptimizer] = None,
        prefix: str = "checkpoint_",
        **kwargs
    ) -> Union[List[PosteriorState], PosteriorState]:
        def _get(_i):
            return self.state[_i].get(
                checkpoint_path=checkpoint_path,
                optimizer=optimizer,
                prefix=prefix,
                **kwargs
            )

        if i is not None:
            return _get(i)
        state = []
        for i in range(self.ensemble_size):
            state.append(_get(i))
        return state

    def put(
        self,
        state: PosteriorState,
        i: int = None,
        checkpoint_path: Optional[Path] = None,
        keep: int = 1,
        prefix: str = "checkpoint_",
    ) -> None:
        def _put(_i):
            return self.state[_i].put(
                state=state, checkpoint_path=checkpoint_path, keep=keep, prefix=prefix
            )

        if i is not None:
            _put(i)
        else:
            for i in range(self.ensemble_size):
                state.append(_put(i))

    def pull(
        self,
        i: int = None,
        checkpoint_path: Path = None,
        optimizer: Optional[OptaxOptimizer] = None,
        prefix: str = "checkpoint_",
        **kwargs
    ) -> Union[DeepEnsembleState, PosteriorState]:
        def _pull(_i):
            return self.state[_i].pull(
                checkpoint_path=checkpoint_path,
                optimizer=optimizer,
                prefix=prefix,
                **kwargs
            )

        if i is not None:
            return _pull(i)
        state = []
        for i in range(self.ensemble_size):
            state.append(_pull(i))
        return state

    def update(
        self,
        variables: Dict,
        i: int = None,
        checkpoint_path: Path = None,
        optimizer: Optional[OptaxOptimizer] = None,
        keep: int = 1,
        prefix: str = "checkpoint_",
        **kwargs
    ):
        def _update(_i):
            self.state[_i].update(
                variables=variables,
                checkpoint_path=checkpoint_path,
                optimizer=optimizer,
                keep=keep,
                prefix=prefix,
                **kwargs
            )

        if i is not None:
            _update(i)
        else:
            for i in range(self.ensemble_size):
                _update(i)

    def extract(
        self,
        keys: List[str],
        i: int = None,
        checkpoint_path: Optional[Path] = None,
        prefix: str = "checkpoint_",
        **kwargs
    ) -> Union[Dict, List[Dict]]:
        def _extract(_i):
            return self.state[_i].extract(
                keys=keys, checkpoint_path=checkpoint_path, prefix=prefix, **kwargs
            )

        if i is not None:
            return _extract(i)
        dicts = []
        for i in range(self.ensemble_size):
            dicts.append(_extract(i))
        return dicts

    def extract_calib_keys(
        self,
        checkpoint_path: Optional[Path] = None,
        prefix: str = "checkpoint_",
        **kwargs
    ) -> Dict:
        return self.extract(
            ["calib_params", "calib_keys"], 0, checkpoint_path, prefix, **kwargs
        )
