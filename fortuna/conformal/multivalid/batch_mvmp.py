from __future__ import annotations

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

from fortuna.conformal.multivalid.base import (
    Group,
    MultivalidMethod,
    Score,
)
from fortuna.data.loader import (
    DataLoader,
    InputsLoader,
)
from fortuna.typing import Array

#
# class BatchMVMPMethod(MultivalidMethod):
#     def __init__(
#         self,
#         score_fn: Optional[Callable[[Array, Array], Array]] = None,
#         group_fns: Optional[List[Callable[[Array], Array]]] = None,
#         model_fn: Optional[Callable[[Array], Array]] = None
#     ):
#         super().__init__(
#             group_fns=group_fns,
#             score_fn=score_fn,
#             model_fn=model_fn,
#         )
#         self.score_fn = Score(score_fn) if score_fn is not None else self._init_missing_score_fn()
#         self.group_fns = [Group(g) for g in group_fns] if group_fns is not None else self._init_missing_group_fns()
#         self.model_fn = model_fn if model_fn is not None else self._init_missing_model_fn()
#
#         self._patch_list = []
#
#     @staticmethod
#     def _init_missing_score_fn():
#         return Score(lambda x, y: y)
#
#     def calibrate(
#         self,
#         val_data_loader: DataLoader,
#         test_inputs_loader: InputsLoader,
#         tol: float = 1e-4,
#         n_rounds: int = 1000,
#         return_errors: bool = False,
#     ) -> Union[Array, Tuple[Array, List[Array]]]:
#         """
#         Compute a threshold :math:`f(x)` of the score functions :math:`s(x,y)` for each test input :math:`x`.
#         Given these threshold, conformal sets can be formulated as :math:`C(x) = \{y: s(x,y) \le f(x)\}`.
#
#         Parameters
#         ----------
#         val_data_loader: DataLoader
#             A data loader of validation data.
#         test_inputs_loader: InputsLoader
#             A loader of test input data points.
#         tol: float
#             A tolerance for the maximum calibration error.
#         n_rounds: int
#             The maximum number of updates the algorithm will run for.
#         return_errors: bool
#             Whether to return a list of computed maximum calibration error, that is the larger calibration error
#             over the different groups.
#
#         Returns
#         -------
#         Union[Array, Tuple[Array, List[Array]]]
#             The compute threshold of the score function for each test input.
#         """
#         if tol >= 1:
#             raise ValueError("`tol` must be smaller than 1.")
#         n_buckets = int(jnp.ceil(1 / tol))
#         buckets = jnp.linspace(0, 1, n_buckets + 1)
#         n_buckets = n_buckets + 1
#
#         values, groups, scores = _compute_utils_over_loader(val_data_loader, self.group_fns, self.score_fn)
#         test_values, test_groups = _compute_utils_over_loader(test_inputs_loader, self.group_fns)
#
#         normalizer = Normalizer(jnp.min(scores), jnp.max(scores))
#         scores = normalizer.normalize(scores)
#
#         n_groups = groups.shape[1]
#
#         def compute_expectation(
#                 v: Array, g: Array, return_prob_b: bool = False
#         ):
#             b = (jnp.abs(values - v) < 0.5 / n_buckets) * groups[:, g]
#             filtered_scores = scores * b
#             prob_b = jnp.mean(b)
#             mean = jnp.where(prob_b > 0, jnp.mean(filtered_scores) / prob_b, 0.0)
#             if return_prob_b:
#                 return mean, prob_b
#             return mean
#
#         def compute_expectation_error(
#             v: Array, g: Array, return_prob_b: bool = False
#         ):
#             if return_prob_b:
#                 mean, prob_b = compute_expectation(v, g, return_prob_b)
#                 return (v - mean) ** 2, prob_b
#             mean = compute_expectation(v, g, return_prob_b)
#             return (v - mean) ** 2
#
#         def calibration_error(v, g):
#             expectation_error, prob_b = compute_expectation_error(v, g, return_prob_b=True)
#             return prob_b * expectation_error
#
#         max_calib_errors = None
#         if return_errors:
#             max_calib_errors = []
#
#         for t in range(n_rounds):
#             calib_error_vg = vmap(
#                 lambda g: vmap(lambda v: calibration_error(v, g))(buckets)
#             )(jnp.arange(n_groups))
#             max_calib_error = calib_error_vg.sum(1).max()
#             if return_errors:
#                 max_calib_errors.append(max_calib_error)
#             if max_calib_error <= tol:
#                 logging.info(
#                     f"After {t} rounds, the algorithm produced mean multicalibrated scores with maximum average "
#                     f"squared calibration error over the groups of {max_calib_error}."
#                 )
#                 break
#
#             gt, idx_vt = jnp.unravel_index(
#                 jnp.argmax(calib_error_vg), (n_groups, n_buckets)
#             )
#             vt = buckets[idx_vt]
#
#             vtildet = compute_expectation(vt, gt)
#             vtildet1 = buckets[jnp.argmin(jnp.abs(vtildet - buckets))]
#
#             bt = (jnp.abs(values - vt) < 0.5 / n_buckets) * groups[:, gt]
#             values = values.at[bt].set(
#                 jnp.minimum(
#                     vtildet1,
#                     jnp.ones_like(values[bt]),
#                 )
#             )
#             test_bt = (
#                 jnp.abs(test_values - vt) < 0.5 / n_buckets
#             ) * test_groups[:, gt]
#             test_values = test_values.at[test_bt].set(
#                 jnp.minimum(
#                     vtildet1,
#                     jnp.ones_like(test_values[test_bt]),
#                 )
#             )
#
#         test_values = normalizer.unnormalize(test_values)
#         if return_errors:
#             return test_values, max_calib_errors
#         return test_values


class BatchMVMPMethod(MultivalidMethod):
    def __init__(
        self,
        score_fn: Optional[Callable[[Array, Array], Array]] = None,
        group_fns: Optional[List[Callable[[Array], Array]]] = None,
        model_fn: Optional[Callable[[Array], Array]] = None,
    ):
        super().__init__(
            group_fns=group_fns,
            score_fn=score_fn,
            model_fn=model_fn,
        )
        self.score_fn = (
            Score(score_fn) if score_fn is not None else self._init_missing_score_fn()
        )
        self.group_fns = (
            [Group(g) for g in group_fns]
            if group_fns is not None
            else self._init_missing_group_fns()
        )
        self.model_fn = (
            model_fn if model_fn is not None else self._init_missing_model_fn()
        )

        self._patch_list = []

    @staticmethod
    def _init_missing_score_fn():
        return Score(lambda x, y: y)

    def calibration_error(
        self,
        v: Array,
        g: Array,
        scores: Array,
        groups: Array,
        values: Array,
        n_buckets: int,
    ):
        expectation_error, prob_b = self._compute_expectation_error(
            v=v,
            g=g,
            scores=scores,
            groups=groups,
            values=values,
            n_buckets=n_buckets,
            return_prob_b=True,
        )
        return prob_b * expectation_error

    def _compute_expectation_error(
        self,
        v: Array,
        g: Array,
        scores: Array,
        groups: Array,
        values: Array,
        n_buckets: int,
        return_prob_b: bool = False,
    ):
        if return_prob_b:
            mean, prob_b = self._compute_expectation(
                v=v,
                g=g,
                scores=scores,
                groups=groups,
                values=values,
                n_buckets=n_buckets,
                return_prob_b=return_prob_b,
            )
            return (v - mean) ** 2, prob_b
        mean = self._compute_expectation(
            v=v,
            g=g,
            scores=scores,
            groups=groups,
            values=values,
            n_buckets=n_buckets,
            return_prob_b=return_prob_b,
        )
        return (v - mean) ** 2

    @staticmethod
    def _compute_expectation(
        v: Array,
        g: Array,
        scores: Array,
        groups: Array,
        values: Array,
        n_buckets: int,
        return_prob_b: bool = False,
    ):
        b = (jnp.abs(values - v) < 0.5 / n_buckets) * groups[:, g]
        filtered_scores = scores * b
        prob_b = jnp.mean(b)
        mean = jnp.where(prob_b > 0, jnp.mean(filtered_scores) / prob_b, 0.0)

        if return_prob_b:
            return mean, prob_b
        return mean

    def _get_patch(
        self,
        vt: Array,
        gt: Array,
        scores: Array,
        groups: Array,
        values: Array,
        buckets: Array,
    ) -> Array:
        patch = self._compute_expectation(
            v=vt,
            g=gt,
            scores=scores,
            groups=groups,
            values=values,
            n_buckets=len(buckets),
        )
        return buckets[jnp.argmin(jnp.abs(patch - buckets))]
