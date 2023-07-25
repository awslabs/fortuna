from __future__ import annotations

import abc
import logging
from typing import (
    Callable,
    List,
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
from fortuna.conformal.multivalid.base import Score, Normalizer, Group, _compute_utils_over_loader


class BatchMVMPMethod(abc.ABC):
    def __init__(
        self,
        score_fn: Callable[[Array, Array], Array],
        group_fns: List[Callable[[Array], Array]],
    ):
        super().__init__()
        self.score_fn = Score(score_fn)
        self.group_fns = [Group(g) for g in group_fns]

    def calibrate(
        self,
        val_data_loader: DataLoader,
        test_inputs_loader: InputsLoader,
        tol: float = 1e-4,
        n_rounds: int = 1000,
        return_max_calib_error: bool = False,
    ) -> Union[Array, Tuple[Array, List[Array]]]:
        """
        Compute a threshold :math:`f(x)` of the score functions :math:`s(x,y)` for each test input :math:`x`.
        Given these threshold, conformal sets can be formulated as :math:`C(x) = \{y: s(x,y) \le f(x)\}`.

        Parameters
        ----------
        val_data_loader: DataLoader
            A data loader of validation data.
        test_inputs_loader: InputsLoader
            A loader of test input data points.
        tol: float
            A tolerance for the maximum calibration error.
        n_rounds: int
            The maximum number of updates the algorithm will run for.
        return_max_calib_error: bool
            Whether to return a list of computed maximum calibration error, that is the larger calibration error
            over the different groups.

        Returns
        -------
        Union[Array, Tuple[Array, List[Array]]]
            The compute threshold of the score function for each test input.
        """
        if tol >= 1:
            raise ValueError("`tol` must be smaller than 1.")
        n_buckets = int(jnp.ceil(1 / tol))
        buckets = jnp.linspace(0, 1, n_buckets + 1)
        n_buckets = n_buckets + 1

        thresholds, groups, scores = _compute_utils_over_loader(val_data_loader, self.group_fns, self.score_fn)
        test_thresholds, test_groups = _compute_utils_over_loader(test_inputs_loader, self.group_fns)

        normalizer = Normalizer(jnp.min(scores), jnp.max(scores))
        scores = normalizer.normalize(scores)

        n_groups = groups.shape[1]

        def compute_expectation(
                v: Array, g: Array, return_prob_b: bool = False
        ):
            b = (jnp.abs(thresholds - v) < 0.5 / n_buckets) * groups[:, g]
            filtered_scores = scores * b
            prob_b = jnp.mean(b)
            mean = jnp.where(prob_b > 0, jnp.mean(filtered_scores) / prob_b, 0.0)
            if return_prob_b:
                return mean, prob_b
            return mean

        def compute_expectation_error(
            v: Array, g: Array, return_prob_b: bool = False
        ):
            if return_prob_b:
                mean, prob_b = compute_expectation(v, g, return_prob_b)
                return (v - mean) ** 2, prob_b
            mean = compute_expectation(v, g, return_prob_b)
            return (v - mean) ** 2

        def calibration_error(v, g):
            expectation_error, prob_b = compute_expectation_error(v, g, return_prob_b=True)
            return prob_b * expectation_error

        max_calib_errors = None
        if return_max_calib_error:
            max_calib_errors = []

        for t in range(n_rounds):
            calib_error_vg = vmap(
                lambda g: vmap(lambda v: calibration_error(v, g))(buckets)
            )(jnp.arange(n_groups))
            max_calib_error = calib_error_vg.sum(1).max()
            if return_max_calib_error:
                max_calib_errors.append(max_calib_error)
            if max_calib_error <= tol:
                logging.info(
                    f"After {t} rounds, the algorithm produced mean multicalibrated scores with maximum average "
                    f"squared calibration error over the groups of {max_calib_error}."
                )
                break

            gt, idx_vt = jnp.unravel_index(
                jnp.argmax(calib_error_vg), (n_groups, n_buckets)
            )
            vt = buckets[idx_vt]

            vtildet = compute_expectation(vt, gt)
            vtildet1 = buckets[jnp.argmin(jnp.abs(vtildet - buckets))]

            bt = (jnp.abs(thresholds - vt) < 0.5 / n_buckets) * groups[:, gt]
            thresholds = thresholds.at[bt].set(
                jnp.minimum(
                    vtildet1,
                    jnp.ones_like(thresholds[bt]),
                )
            )
            test_bt = (
                jnp.abs(test_thresholds - vt) < 0.5 / n_buckets
            ) * test_groups[:, gt]
            test_thresholds = test_thresholds.at[test_bt].set(
                jnp.minimum(
                    vtildet1,
                    jnp.ones_like(test_thresholds[test_bt]),
                )
            )

        test_thresholds = normalizer.unnormalize(test_thresholds)
        if return_max_calib_error:
            return test_thresholds, max_calib_errors
        return test_thresholds
