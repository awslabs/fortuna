import jax.numpy as jnp
from typing import Callable, Optional, Union, List
from fortuna.typing import Array
from fortuna.data.loader import DataLoader, InputsLoader


class Group:
    def __init__(self, group_fn: Callable[[Array], Array]):
        self.group_fn = group_fn

    def __call__(self, x):
        g = self.group_fn(x)
        if g.ndim > 1:
            raise ValueError(
                "Evaluations of the group function `group_fn` must be one-dimensional arrays."
            )
        if jnp.any((g != 0) * (g != 1)):
            raise ValueError(
                "The group function `threshold_fn` must take values in {0, 1}."
            )
        return g.astype(bool)


class Normalizer:
    def __init__(self, xmin: Array, xmax: Array):
        self.xmin = xmin
        self.xmax = xmax if xmax != xmin else xmin + 1

    def normalize(self, x: Array) -> Array:
        return (x - self.xmin) / (self.xmax - self.xmin)

    def unnormalize(self, y: Array) -> Array:
        return self.xmin + (self.xmax - self.xmin) * y


class Score:
    def __init__(self, score_fn: Callable[[Array, Array], Array]):
        self.score_fn = score_fn

    def __call__(self, x: Array, y: Array):
        s = self.score_fn(x, y)
        if s.ndim > 1:
            raise ValueError(
                "Evaluations of the score function `score_fn` must be one-dimensional arrays, "
                f"but its shape was {s.shape}."
            )
        return s


def _compute_utils_over_loader(
        loader: Union[DataLoader, InputsLoader],
        group_fns: List[Group] = None,
        score_fn: Optional[Score] = None
):
    thresholds, groups = [], []
    if score_fn:
        scores = []

    for batch in loader:
        if score_fn:
            inputs, targets = batch
            scores.append(score_fn(inputs, targets))
        else:
            inputs = batch
        thresholds.append(jnp.zeros(inputs.shape[0]))
        groups.append(
            jnp.concatenate([g(inputs)[:, None] for g in group_fns], axis=1)
        )
    thresholds, groups = (
        jnp.concatenate(thresholds),
        jnp.concatenate(groups, 0),
    )
    if score_fn:
        scores = jnp.concatenate(scores)
        return thresholds, groups, scores
    return thresholds, groups
