from jax import vmap
import jax.numpy as jnp

from fortuna.conformal.classification.base import SplitConformalClassifier, CVPlusConformalClassifier
from fortuna.typing import Array


@vmap
def _score_fn(probs: Array, perm: Array, inv_perm: Array, targets: Array):
    return jnp.cumsum(probs[perm])[inv_perm[targets]]


def score_fn(
        probs: Array,
        targets: Array,
):
    perms = jnp.argsort(probs, axis=1)[:, ::-1]
    inv_perms = jnp.argsort(perms, axis=1)
    return _score_fn(probs, perms, inv_perms, targets)


class AdaptivePredictionConformalClassifier(SplitConformalClassifier):
    def score_fn(
            self,
            probs: Array,
            targets: Array,
    ):
        return score_fn(probs=probs, targets=targets)


class CVPlusAdaptivePredictionConformalClassifier(CVPlusConformalClassifier):
    def score_fn(
            self,
            probs: Array,
            targets: Array,
    ):
        return score_fn(probs=probs, targets=targets)
