from jax import vmap

from fortuna.conformal.classification.base import (
    CVPlusConformalClassifier,
    SplitConformalClassifier,
)
from fortuna.typing import Array


@vmap
def _score_fn(probs: Array, target: Array):
    return 1 - probs[target]


def score_fn(
    probs: Array,
    targets: Array,
):
    return _score_fn(probs, targets)


class SimplePredictionConformalClassifier(SplitConformalClassifier):
    def score_fn(
        self,
        probs: Array,
        targets: Array,
    ):
        return score_fn(probs=probs, targets=targets)


class CVPlusSimplePredictionConformalClassifier(CVPlusConformalClassifier):
    def score_fn(
        self,
        probs: Array,
        targets: Array,
    ):
        return score_fn(probs=probs, targets=targets)
