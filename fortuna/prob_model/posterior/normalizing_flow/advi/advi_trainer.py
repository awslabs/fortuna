from fortuna.prob_model.posterior.normalizing_flow.advi import ADVI_NAME
from fortuna.prob_model.posterior.normalizing_flow.normalizing_flow_trainer import \
    NormalizingFlowTrainer
from fortuna.utils.nested_dicts import nested_set, nested_pair
from fortuna.prob_model.posterior.normalizing_flow.advi.advi_state import ADVIState
from fortuna.typing import Path, Array, Params
from typing import Optional, List
import jax.numpy as jnp
from flax.core import FrozenDict


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
        state = state.replace(params=self._get_mean_std_from_rav_mean_logvar(state.params, self._all_params))
        super().save_checkpoint(
            state,
            save_checkpoint_dir,
            keep,
            force_save,
            prefix
        )

    def _get_mean_std_from_rav_mean_logvar(
            self,
            mean_logvar_rav: FrozenDict,
            all_params: Optional[Params] = None,
    ) -> Params:
        if self._which_params is not None:
            means = tuple([_unravel(mean_logvar_rav["mean"][self._idx[i]:self._idx[i + 1]]) for i, _unravel in enumerate(self._unravel)])
            stds = tuple([_unravel(jnp.exp(0.5 * mean_logvar_rav["logvar"][self._idx[i]:self._idx[i + 1]])) for i, _unravel in enumerate(self._unravel)])

            all_params = all_params.unfreeze()
            all_params = nested_set(all_params, self._which_params, means)
            all_params = nested_pair(
                all_params,
                self._which_params,
                stds,
                ("mean", "std"),
            )
            return FrozenDict(all_params)

        params = self._unravel(mean_logvar_rav["mean"]).unfreeze()
        stds = self._unravel(jnp.exp(0.5 * mean_logvar_rav["logvar"]))
        for k, v in params.items():
            params[k] = {"params": dict(mean=v["params"], std=stds[k]["params"])}
        return FrozenDict(params)

    def on_train_end(self, state: ADVIState) -> ADVIState:
        self.save_checkpoint(
            state,
            save_checkpoint_dir=self.save_checkpoint_dir,
            keep=self.keep_top_n_checkpoints,
            force_save=True,
        )
        return state.replace(params=self._get_mean_std_from_rav_mean_logvar(state.params, self._all_params))
