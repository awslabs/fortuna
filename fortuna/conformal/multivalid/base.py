import abc
import logging
from typing import (
    Callable,
    List,
    Optional,
    Tuple,
    Union,
)

from jax import vmap
import jax.numpy as jnp

from fortuna.data.loader import (
    DataLoader,
    InputsLoader,
)
from fortuna.typing import Array


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
    def __init__(self, x_min: Array, x_max: Array):
        self.x_min = x_min
        self.x_max = x_max if x_max != x_min else x_min + 1

    def normalize(self, x: Array) -> Array:
        return (x - self.x_min) / (self.x_max - self.x_min)

    def unnormalize(self, y: Array) -> Array:
        return self.x_min + (self.x_max - self.x_min) * y


class Score:
    def __init__(self, score_fn: Callable[[Array, Array], Array]):
        self.score_fn = score_fn

    def __call__(self, x: Array, y: Array) -> Array:
        s = self.score_fn(x, y)
        if s.ndim > 1:
            raise ValueError(
                "Evaluations of the score function `score_fn` must be one-dimensional arrays, "
                f"but its shape was {s.shape}."
            )
        return s


class Model:
    def __init__(self, model_fn: Callable[[Array], Array]):
        self.model_fn = model_fn

    def __call__(self, x: Array):
        v = self.model_fn(x)
        if v.ndim > 1:
            raise ValueError(
                "Evaluations of the model function `model_fn` must be one-dimensional arrays, "
                f"but its shape was {v.shape}."
            )
        if jnp.any(v < 0) or jnp.any(v > 1):
            raise ValueError("The model function must take values within [0, 1].")
        return v


class MultivalidMethod:
    def __init__(
        self,
        group_fns: List[Callable[[Array], Array]],
        score_fn: Callable[[Array, Array], Array],
        model_fn: Callable[[Array], Array],
    ):
        self.group_fns = [Group(g) for g in group_fns]
        self.score_fn = Score(score_fn)
        self.model_fn = Model(model_fn)

        self._patch_fns = []
        self._normalizer = None

    def calibrate(
        self,
        calib_data_loader: DataLoader,
        test_inputs_loader: Optional[InputsLoader] = None,
        tol: float = 1e-4,
        n_rounds: int = 1000,
    ) -> Union[List[Array], Tuple[Array, List[Array]]]:
        if tol >= 1:
            raise ValueError("`tol` must be smaller than 1.")
        n_buckets = int(jnp.ceil(1 / tol)) + 1
        buckets = jnp.linspace(0, 1, n_buckets)

        (
            scores,
            groups,
            values,
        ) = self._eval_scores_and_groups_and_values_over_data_loader(calib_data_loader)
        if test_inputs_loader is not None:
            (
                test_groups,
                test_values,
            ) = self._eval_groups_and_and_values_over_inputs_loader(test_inputs_loader)

        self._normalizer = Normalizer(jnp.min(scores), jnp.max(scores))
        scores = self._normalizer.normalize(scores)

        n_groups = groups.shape[1]

        max_calib_errors = []

        for t in range(n_rounds):
            calib_error_vg = vmap(
                lambda g: vmap(
                    lambda v: self.calibration_error(
                        v,
                        g,
                        scores=scores,
                        groups=groups,
                        values=values,
                        n_buckets=n_buckets,
                    )
                )(buckets)
            )(jnp.arange(n_groups))

            max_calib_errors.append(calib_error_vg.sum(1).max())
            if max_calib_errors[-1] <= tol:
                logging.info(f"Tolerance satisfied after {t} rounds.")
                break

            gt, vt = self._get_gt_and_vt(
                calib_error_vg=calib_error_vg, buckets=buckets, n_groups=n_groups
            )
            self.patch(
                scores=scores,
                groups=groups,
                values=values,
                vt=vt,
                gt=gt,
                buckets=buckets,
            )

            self._patch_fns.append(
                lambda _groups, _values: self.patch(
                    scores=scores,
                    groups=_groups,
                    values=_values,
                    vt=vt,
                    gt=gt,
                    buckets=buckets,
                )
            )
            if test_inputs_loader is not None:
                test_values = self._patch_fns[-1](test_groups, test_values)

        if test_inputs_loader is not None:
            test_values = self._normalizer.unnormalize(test_values)
            return test_values, max_calib_errors
        return max_calib_errors

    def apply_patches(
        self,
        inputs_loader: InputsLoader,
    ) -> Array:
        if not len(self._patch_fns):
            raise ValueError(
                "No patch available. Please make sure to run `calibrate` first."
            )

        groups, values = self._eval_groups_and_and_values_over_inputs_loader(
            inputs_loader
        )
        for patch_fn in self._patch_fns:
            values = patch_fn(groups, values)
        values = self._normalizer.unnormalize(values)
        return values

    @abc.abstractmethod
    def calibration_error(
        self,
        v: Array,
        g: Array,
        scores: Array,
        groups: Array,
        values: Array,
        n_buckets: int,
    ):
        pass

    @staticmethod
    def _init_missing_group_fns():
        return [Group(lambda x: jnp.ones(x.shape[0], dtype=bool))]

    @staticmethod
    def _init_missing_model_fn():
        return Model(lambda x: jnp.zeros(x.shape[0]))

    def _eval_scores_and_groups_and_values_over_data_loader(
        self, data_loader: DataLoader
    ) -> Tuple[Array, Array, Array]:
        scores, groups, values = [], [], []

        for inputs, targets in data_loader:
            scores.append(self.score_fn(inputs, targets))
            groups.append(
                jnp.concatenate([g(inputs)[:, None] for g in self.group_fns], axis=1)
            )
            values.append(self.model_fn(inputs))

        scores = jnp.concatenate(scores)
        groups = jnp.concatenate(groups, 0)
        values = jnp.concatenate(values)

        return scores, groups, values

    def _eval_groups_and_and_values_over_inputs_loader(
        self, inputs_loader: InputsLoader
    ) -> Tuple[Array, Array]:
        groups, values = [], []

        for inputs in inputs_loader:
            groups.append(
                jnp.concatenate([g(inputs)[:, None] for g in self.group_fns], axis=1)
            )
            values.append(self.model_fn(inputs))

        groups = jnp.concatenate(groups, 0)
        values = jnp.concatenate(values)

        return groups, values

    @staticmethod
    def _get_gt_and_vt(
        calib_error_vg: Array, buckets: Array, n_groups: int
    ) -> Tuple[Array, Array]:
        gt, idx_vt = jnp.unravel_index(
            jnp.argmax(calib_error_vg), (n_groups, len(buckets))
        )
        vt = buckets[idx_vt]
        return gt, vt

    @staticmethod
    def _get_bt(
        groups: Array, values: Array, vt: Array, gt: Array, n_buckets: int
    ) -> Array:
        return (jnp.abs(values - vt) < 0.5 / n_buckets) * groups[:, gt]

    @abc.abstractmethod
    def _get_patch(
        self,
        vt: Array,
        gt: Array,
        scores: Array,
        groups: Array,
        values: Array,
        buckets: Array,
    ) -> Array:
        pass

    @staticmethod
    def _patch(values: Array, patch: Array, bt: Array) -> Array:
        return values.at[bt].set(
            jnp.minimum(
                patch,
                jnp.ones_like(values[bt]),
            )
        )

    def patch(
        self,
        scores: Array,
        groups: Array,
        values: Array,
        vt: Array,
        gt: Array,
        buckets: Array,
    ) -> Array:
        bt = self._get_bt(
            groups=groups, values=values, vt=vt, gt=gt, n_buckets=len(buckets)
        )
        patch = self._get_patch(
            vt=vt, gt=gt, scores=scores, groups=groups, values=values, buckets=buckets
        )
        return self._patch(values=values, patch=patch, bt=bt)


def _compute_utils_over_loader(
    loader: Union[DataLoader, InputsLoader],
    group_fns: List[Group] = None,
    score_fn: Optional[Score] = None,
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
        groups.append(jnp.concatenate([g(inputs)[:, None] for g in group_fns], axis=1))
    thresholds, groups = (
        jnp.concatenate(thresholds),
        jnp.concatenate(groups, 0),
    )
    if score_fn:
        scores = jnp.concatenate(scores)
        return thresholds, groups, scores
    return thresholds, groups
