from fortuna.prob_model.posterior.normalizing_flow.advi import ADVI_NAME
from fortuna.prob_model.posterior.normalizing_flow.normalizing_flow_trainer import \
    NormalizingFlowTrainer
from fortuna.utils.nested_dicts import nested_set, nested_pair
from fortuna.prob_model.posterior.normalizing_flow.advi.advi_state import ADVIState
from fortuna.typing import Path, Params
from typing import Optional
from flax.core import FrozenDict
import jax
from fortuna.training.trainer import MultiDeviceMixin, JittedMixin
from jax.tree_util import tree_map


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
        state = state.replace(params=self._unravel_params(state.params, self._all_params))
        super().save_checkpoint(
            state,
            save_checkpoint_dir,
            keep,
            force_save,
            prefix
        )

    def _unravel_params(
            self,
            rav_params: FrozenDict,
            all_params: Optional[Params] = None,
    ) -> Params:
        if self._which_params is not None:
            means = tuple([_unravel(
                rav_params["mean"][self._indices[i]:self._indices[i + 1]]) for i, _unravel in enumerate(self._unravel)])
            log_stds = tuple([_unravel(rav_params["log_std"][self._indices[i]:self._indices[i + 1]]) for i, _unravel in enumerate(self._unravel)])

            all_params = all_params.unfreeze()
            all_params = nested_set(all_params, self._which_params, means)
            all_params = nested_pair(
                all_params,
                self._which_params,
                log_stds,
                ("mean", "log_std"),
            )
            return FrozenDict(all_params)

        params = self._unravel(rav_params["mean"]).unfreeze()
        log_stds = self._unravel(rav_params["log_std"])
        for k, v in params.items():
            params[k] = {"params": dict(mean=v["params"], log_std=log_stds[k]["params"])}
        return FrozenDict(params)

    def on_train_end(self, state: ADVIState) -> ADVIState:
        self.save_checkpoint(
            state,
            save_checkpoint_dir=self.save_checkpoint_dir,
            keep=self.keep_top_n_checkpoints,
            force_save=True,
        )
        return state.replace(params=self._unravel_params(state.params, self._all_params))


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
        return state.replace(params=self._unravel_params(state.params, self._all_params))
