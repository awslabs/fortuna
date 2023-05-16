from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from flax.core import FrozenDict
from fortuna.prob_model.posterior.posterior_multi_state_repository import (
    PosteriorMultiStateRepository,
)
from fortuna.prob_model.posterior.state import PosteriorState
from fortuna.typing import (
    AnyKey,
    OptaxOptimizer,
    Path,
    Params,
)
from fortuna.utils.nested_dicts import (
    nested_get,
    nested_set,
)


class SGMCMCPosteriorStateRepository(PosteriorMultiStateRepository):
    def __init__(self,
                 size: int,
                 checkpoint_dir: Optional[Path] = None,
                 all_params: Optional[Params] = None,
                 which_params: Optional[Tuple[List[AnyKey], ...]] = None):
        super().__init__(size=size, checkpoint_dir=checkpoint_dir)
        self._all_params = all_params
        self._which_params = which_params

    def get(
        self,
        i: int = None,
        checkpoint_path: Optional[Path] = None,
        optimizer: Optional[OptaxOptimizer] = None,
        prefix: str = "checkpoint_",
        **kwargs,
    ) -> Union[List[PosteriorState], PosteriorState]:
        state = super().get(
            i=i,
            checkpoint_path=checkpoint_path,
            optimizer=optimizer,
            prefix=prefix,
            **kwargs,
        )
        return self._update_state(state, modify="add")

    def put(
        self,
        state: PosteriorState,
        i: int = None,
        checkpoint_path: Optional[Path] = None,
        keep: int = 1,
        prefix: str = "checkpoint_",
    ) -> None:
        state = self._update_state(state, modify="remove")
        return super().put(
            state=state,
            i=i,
            checkpoint_path=checkpoint_path,
            keep=keep,
            prefix=prefix,
        )

    def pull(
        self,
        i: int = None,
        checkpoint_path: Path = None,
        optimizer: Optional[OptaxOptimizer] = None,
        prefix: str = "checkpoint_",
        **kwargs,
    ) -> PosteriorState:
        state = super().pull(
            i=i,
            checkpoint_path=checkpoint_path,
            optimizer=optimizer,
            prefix=prefix,
            **kwargs
        )
        return self._update_state(state, modify="add")

    def extract(
        self,
        keys: List[str],
        i: int = None,
        checkpoint_path: Optional[Path] = None,
        prefix: str = "checkpoint_",
        **kwargs,
    ) -> Union[Dict, List[Dict]]:
        def _extract(_i):
            state = self.get(
                i=_i,
                checkpoint_path=checkpoint_path,
                prefix=prefix,
            )
            return {k: getattr(state, k) for k in keys}

        if i is not None:
            return _extract(i)
        dicts = []
        for i in range(self.size):
            dicts.append(_extract(i))
        return dicts

    def _update_state(
        self,
        state: Union[List[PosteriorState], PosteriorState],
        modify: str = "add",
    ) -> Union[List[PosteriorState], PosteriorState]:
        if self._which_params is None:
            return state

        if isinstance(state, list):
            return [_update_state(_state, modify=modify) for _state in state]

        if modify == "add":
            state = state.replace(
                params=FrozenDict(
                    nested_set(
                        d=self._all_params.unfreeze(),
                        key_paths=self._which_params,
                        objs=tuple(
                            [
                                nested_get(d=state.params, keys=path)
                                for path in self._which_params
                            ]
                        ),
                    )
                )
            )
        elif modify == "remove":
            state = state.replace(
                params=FrozenDict(
                    nested_set(
                        d={},
                        key_paths=self._which_params,
                        objs=tuple(
                            [
                                nested_get(d=state.params, keys=path)
                                for path in self._which_params
                            ]
                        ),
                        allow_nonexistent=True,
                    )
                ),
                step=state.step,
            )

        return state
