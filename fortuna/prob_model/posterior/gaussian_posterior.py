import abc

from fortuna.prob_model.posterior.base import Posterior
from typing import Optional
from flax.core import FrozenDict
from jax._src.prng import PRNGKeyArray
from fortuna.prob_model.joint.state import JointState
from fortuna.utils.nested_dicts import nested_set, nested_unpair, nested_get
from jax.tree_util import tree_map
from fortuna.utils.random import generate_random_normal_like_tree


class GaussianPosterior(Posterior, abc.ABC):
    def _sample_diag_gaussian_from_mean_and_std(self, rng: Optional[PRNGKeyArray] = None) -> JointState:
        if rng is None:
            rng = self.rng.get()
        state = self.state.get()

        if hasattr(self.posterior_approximator, "which_params") and self.posterior_approximator.which_params is not None:
            mean, std = nested_unpair(
                state.params.unfreeze(),
                self.posterior_approximator.which_params,
                ("mean", "std"),
            )

            noise = generate_random_normal_like_tree(rng, std)
            params = nested_set(
                d=mean,
                key_paths=self.posterior_approximator.which_params,
                objs=tuple(
                    [
                        tree_map(
                            lambda m, s, e: m + s * e,
                            nested_get(mean, keys),
                            nested_get(std, keys),
                            nested_get(noise, keys),
                        )
                        for keys in self.posterior_approximator.which_params
                    ]
                ),
            )
            for k, v in params.items():
                params[k] = FrozenDict(v)
            state = state.replace(params=FrozenDict(params))
        else:
            mean, std = dict(), dict()
            for k, v in state.params.items():
                mean[k] = FrozenDict({"params": v["params"]["mean"]})
                std[k] = FrozenDict({"params": v["params"]["std"]})

            state = state.replace(
                params=FrozenDict(
                    tree_map(
                        lambda m, s, e: m + s * e,
                        mean,
                        std,
                        generate_random_normal_like_tree(rng, std),
                    )
                )
            )

        return JointState(
            params=state.params,
            mutable=state.mutable,
            calib_params=state.calib_params,
            calib_mutable=state.calib_mutable,
        )
