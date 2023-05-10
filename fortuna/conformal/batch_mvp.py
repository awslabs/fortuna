from __future__ import annotations

import abc
import logging
from typing import Callable, List, Tuple, Union

import jax.numpy as jnp
from jax import vmap

from fortuna.data.loader import DataLoader, InputsLoader
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


class BatchMVPConformalMethod(abc.ABC):
    def __init__(
        self,
        score_fn: Callable[[Array, Array], Array],
        group_fns: List[Callable[[Array], Array]],
        n_buckets: int = 100,
    ):
        super().__init__()
        self.score_fn = Score(score_fn)
        self.group_fns = [Group(g) for g in group_fns]
        self.buckets = jnp.linspace(0, 1, n_buckets + 1)
        self.n_buckets = n_buckets + 1

    def threshold_score(
        self,
        val_data_loader: DataLoader,
        test_inputs_loader: InputsLoader,
        error: float = 0.05,
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
        error: float
            A desired coverage error.
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
        quantile = 1 - error

        scores, thresholds, groups = [], [], []
        for inputs, targets in val_data_loader:
            scores.append(self.score_fn(inputs, targets))
            thresholds.append(jnp.zeros(inputs.shape[0]))
            groups.append(
                jnp.concatenate([g(inputs)[:, None] for g in self.group_fns], axis=1)
            )
        scores, thresholds, groups = (
            jnp.concatenate(scores),
            jnp.concatenate(thresholds),
            jnp.concatenate(groups, 0),
        )

        test_thresholds, test_groups = [], []
        for inputs in test_inputs_loader:
            test_thresholds.append(jnp.zeros(inputs.shape[0]))
            test_groups.append(
                jnp.concatenate([g(inputs)[:, None] for g in self.group_fns], axis=-1)
            )
        test_thresholds, test_groups = jnp.concatenate(
            test_thresholds
        ), jnp.concatenate(test_groups, 0)

        normalizer = Normalizer(jnp.min(scores), jnp.max(scores))
        scores = normalizer.normalize(scores)

        n_groups = groups.shape[1]

        def compute_probability_error(
            v: Array, g: Array, delta: Union[Array, float] = 0.0
        ):
            b = (jnp.abs(thresholds - v) < 0.5 / self.n_buckets) * groups[:, g]
            filtered_scores = jnp.where(b, scores, -jnp.ones_like(scores))
            conds = (filtered_scores <= v + delta) * (filtered_scores != -1)
            prob_b = jnp.mean(b)
            prob = jnp.where(prob_b > 0, jnp.mean(conds) / prob_b, 0.0)
            return (quantile - prob) ** 2

        def calibration_error(v, g):
            b = (jnp.abs(thresholds - v) < 0.5 / self.n_buckets) * groups[:, g]
            filtered_scores = jnp.where(b, scores, -jnp.ones_like(scores))
            conds = (filtered_scores <= v) * (filtered_scores != -1)
            prob_b = jnp.mean(b)
            prob = jnp.where(prob_b > 0, jnp.mean(conds) / prob_b, 0.0)
            return prob_b * (quantile - prob) ** 2

        max_calib_errors = None
        if return_max_calib_error:
            max_calib_errors = []

        for t in range(n_rounds):
            calib_error_vg = vmap(
                lambda g: vmap(lambda v: calibration_error(v, g))(self.buckets)
            )(jnp.arange(n_groups))
            max_calib_error = calib_error_vg.sum(1).max()
            if return_max_calib_error:
                max_calib_errors.append(max_calib_error)
            if max_calib_error <= tol:
                logging.info(
                    f"The algorithm produced a {tol}-approximately {quantile}-quantile multicalibrated "
                    f"threshold function after {t} rounds."
                )
                break

            gt, idx_vt = jnp.unravel_index(
                jnp.argmax(calib_error_vg), (n_groups, self.n_buckets)
            )
            vt = self.buckets[idx_vt]

            deltat = self.buckets[
                jnp.argmin(
                    jnp.abs(
                        vmap(lambda delta: compute_probability_error(vt, gt, delta))(
                            self.buckets
                        )
                    )
                )
            ]
            bt = (jnp.abs(thresholds - vt) < 0.5 / self.n_buckets) * groups[:, gt]
            thresholds = thresholds.at[bt].set(
                jnp.minimum(thresholds[bt] + deltat, jnp.ones_like(thresholds[bt]))
            )
            test_bt = (
                jnp.abs(test_thresholds - vt) < 0.5 / self.n_buckets
            ) * test_groups[:, gt]
            test_thresholds = test_thresholds.at[test_bt].set(
                jnp.minimum(
                    test_thresholds[test_bt] + deltat,
                    jnp.ones_like(test_thresholds[test_bt]),
                )
            )

        test_thresholds = normalizer.unnormalize(test_thresholds)
        if return_max_calib_error:
            return test_thresholds, max_calib_errors
        return test_thresholds
