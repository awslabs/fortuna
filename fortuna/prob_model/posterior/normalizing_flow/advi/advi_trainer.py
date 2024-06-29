from typing import (
    List,
    Tuple,
)

from flax.core import FrozenDict
import jax
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map

from fortuna.data.loader import DataLoader
from fortuna.prob_model.posterior.normalizing_flow.advi import ADVI_NAME
from fortuna.prob_model.posterior.normalizing_flow.advi.advi_state import ADVIState
from fortuna.prob_model.posterior.normalizing_flow.normalizing_flow_state import (
    NormalizingFlowState,
)
from fortuna.prob_model.posterior.normalizing_flow.normalizing_flow_trainer import (
    NormalizingFlowTrainer,
)
from fortuna.training.trainer import (
    JittedMixin,
    MultiDeviceMixin,
)
from fortuna.typing import (
    Params,
    Path,
)
from fortuna.utils.freeze import (
    get_frozen_paths,
    get_trainable_paths,
)
from fortuna.utils.nested_dicts import (
    nested_get,
    nested_pair,
    nested_set,
)


class ADVITrainer(NormalizingFlowTrainer):
    def __str__(self):
        return ADVI_NAME

    def save_checkpoint(
        self,
        state: ADVIState,
        save_checkpoint_dir: Path,
        keep: int = 1,
        force_save: bool = False,
        prefix: str = "checkpoint_",
    ) -> None:
        state = state.replace(
            params=self._unravel_params(state.params),
            frozen_params=FrozenDict(),
            _encoded_which_params=self._encoded_which_params,
        )
        super().save_checkpoint(state, save_checkpoint_dir, keep, force_save, prefix)

    def on_train_end(self, state: NormalizingFlowState) -> NormalizingFlowState:
        self.save_checkpoint(
            state,
            save_checkpoint_dir=self.save_checkpoint_dir,
            keep=self.keep_top_n_checkpoints,
            force_save=True,
        )

        state = state.replace(
            params=self._unravel_params(state.params),
            frozen_params=None,
            _encoded_which_params=self._encoded_which_params,
        )
        return state

    def _unravel_params(
        self,
        rav_params: FrozenDict,
    ) -> Params:
        if self._which_params is not None:
            return FrozenDict(
                nested_pair(
                    self._unravel(rav_params["mean"]).unfreeze(),
                    self._which_params,
                    self._sub_unravel(rav_params["log_std"]),
                    ("mean", "log_std"),
                )
            )

        params = self._unravel(rav_params["mean"]).unfreeze()
        log_stds = self._unravel(rav_params["log_std"])
        for k, v in params.items():
            params[k] = {
                "params": dict(mean=v["params"], log_std=log_stds[k]["params"])
            }
        return FrozenDict(params)

    def on_train_start(
        self,
        state: NormalizingFlowState,
        dataloaders: List[DataLoader],
        rng: jax.Array,
    ) -> Tuple[NormalizingFlowState, List[DataLoader], jax.Array]:
        if self.freeze_fun is not None:
            frozen_paths = get_frozen_paths(
                self._unravel(state.params["mean"]), self.freeze_fun
            )
            trainable_paths = get_trainable_paths(
                self._unravel(state.params["mean"]), self.freeze_fun
            )

            state = state.replace(
                frozen_params=FrozenDict(
                    {
                        s: ravel_pytree(
                            nested_set(
                                d={},
                                key_paths=frozen_paths,
                                objs=tuple(
                                    [
                                        nested_get(
                                            self._unravel(state.params[s]).unfreeze(),
                                            path,
                                        )
                                        for path in frozen_paths
                                    ]
                                ),
                                allow_nonexistent=True,
                            )
                        )[0]
                        for s in ["mean", "log_std"]
                    }
                ),
                params=FrozenDict(
                    {
                        s: ravel_pytree(
                            nested_set(
                                d={},
                                key_paths=trainable_paths,
                                objs=tuple(
                                    [
                                        nested_get(
                                            self._unravel(state.params[s]).unfreeze(),
                                            path,
                                        )
                                        for path in trainable_paths
                                    ]
                                ),
                                allow_nonexistent=True,
                            )
                        )[0]
                        for s in ["mean", "log_std"]
                    }
                ),
            )

        return state, dataloaders, rng


class JittedADVITrainer(JittedMixin, ADVITrainer):
    pass


class MultiDeviceADVITrainer(MultiDeviceMixin, ADVITrainer):
    def on_train_end(self, state: ADVIState) -> ADVIState:
        self.save_checkpoint(
            state,
            save_checkpoint_dir=self.save_checkpoint_dir,
            keep=self.keep_top_n_checkpoints,
            force_save=True,
        )

        state = jax.device_get(tree_map(lambda x: x[0], state))
        state = state.replace(
            params=self._unravel_params(state.params),
            frozen_params=None,
            _encoded_which_params=self._encoded_which_params,
        )
        return state
