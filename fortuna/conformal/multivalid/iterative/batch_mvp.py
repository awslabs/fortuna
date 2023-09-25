from typing import (
    Dict,
    Optional,
    Tuple,
    Union,
)

from jax import vmap
import jax.numpy as jnp

from fortuna.conformal.classification.base import ConformalClassifier
from fortuna.conformal.multivalid.iterative.base import IterativeMultivalidMethod
from fortuna.conformal.multivalid.mixins.batchmvp import BatchMVPMixin
from fortuna.typing import Array


class BatchMVPConformalMethod(
    BatchMVPMixin, IterativeMultivalidMethod, ConformalClassifier
):
    def __init__(self, seed: int = 0):
        """
        This class implements a classification version of BatchMVP
        `[Jung et al., 2022] <https://arxiv.org/abs/2209.15145>`_,
        a multivalid conformal prediction method that satisfies coverage guarantees conditioned on group membership
        and non-conformity threshold.

        Parameters
        ----------
        seed: int
            Random seed.
        """
        super().__init__(seed=seed)
        self._coverage = None

    def calibrate(
        self,
        scores: Array,
        groups: Optional[Array] = None,
        thresholds: Optional[Array] = None,
        test_groups: Optional[Array] = None,
        test_thresholds: Optional[Array] = None,
        atol: float = 1e-4,
        rtol: float = 1e-6,
        min_prob_b: Union[float, str] = "auto",
        n_buckets: int = 100,
        n_rounds: int = 1000,
        eta: float = 0.1,
        split: float = 0.8,
        bucket_types: Tuple[str, ...] = (">=", "<="),
        coverage: float = 0.95,
    ) -> Union[Dict, Tuple[Array, Dict]]:
        """
        Calibrate the model by finding a list of patches to the model that bring the calibration error below a
        certain threshold.

        Parameters
        ----------
        scores: Array
            A list of scores :math:`s(x, y)` computed on the calibration data.
            This should be a one-dimensional array of elements between 0 and 1.
        groups: Array
            A list of groups :math:`g(x)` computed on the calibration data.
            This should be a two-dimensional array of bool elements.
            The first dimension is over the data points, the second dimension is over the number of groups.
        thresholds: Optional[Array]
            The initial model evalutions :math:`f(x)` on the calibration data. If not provided, these are set to 0.
        test_groups: Optional[Array]
            A list of groups :math:`g(x)` computed on the test data.
            This should be a two-dimensional array of bool elements.
            The first dimension is over the data points, the second dimension is over the number of groups.
        test_thresholds: Optional[Array]
            The initial model evaluations :math:`f(x)` on the test data. If not provided, these are set to 0.
        atol: float
            Absolute tolerance on the mean squared error.
        rtol: float
            Relative tolerance on the mean squared error.
        min_prob_b: float
            Minimum probability of the conditioning set :math:`B_t` for the patch to be applied.
        n_buckets: int
            The number of buckets used in the algorithm. The smaller the number of buckets, the simpler the model,
            the better its generalization abilities. If not provided, We start from 2 buckets, and progressively double
            the number of buckets until we find a value for which the calibration error falls below the given
            tolerance. Such number of buckets is guaranteed to exist.
        n_rounds: int
            The maximum number of rounds to run the method for.
        eta: float
            Step size. By default, this is set to 1.
        split: float
            Split the calibration data into calibration and validation, according to the given proportion.
            The validation data will be used for early stopping.
        bucket_types: Tuple[str, ...]
            Types of buckets. The following types are currently supported:

            - "=", corresponding of buckets like :math:`\{f(x) = v\}`;
            - ">=", corresponding of buckets like :math:`\{f(x) \ge v\}`;
            - "<=", corresponding of buckets like :math:`\{f(x) \le v\}`.
        coverage: float
            The desired level of coverage. This must be a scalar between 0 and 1.
        Returns
        -------
        Union[Dict, Tuple[Array, Dict]]
            A status including the number of rounds taken to reach convergence and the calibration errors computed
            during the training procedure. if `test_thresholds` and `test_groups` are provided, the list of patches will
            be applied to `test_thresholds`, and the calibrated test thresholds will be returned together with the status.
        """
        self._check_coverage(coverage)
        self._coverage = coverage
        return super().calibrate(
            scores=scores,
            groups=groups,
            values=thresholds,
            test_groups=test_groups,
            test_values=test_thresholds,
            atol=atol,
            rtol=rtol,
            min_prob_b=min_prob_b,
            n_buckets=n_buckets,
            n_rounds=n_rounds,
            eta=eta,
            split=split,
            bucket_types=bucket_types,
            coverage=coverage,
        )

    def apply_patches(
        self,
        groups: Optional[Array] = None,
        thresholds: Optional[Array] = None,
    ) -> Array:
        return super().apply_patches(groups=groups, values=thresholds)

    def calibration_error(
        self,
        scores: Array,
        groups: Optional[Array] = None,
        thresholds: Optional[Array] = None,
        n_buckets: int = 10000,
        **kwargs,
    ) -> Array:
        return super().calibration_error(
            scores=scores,
            groups=groups,
            values=thresholds,
            n_buckets=n_buckets,
            coverage=self._coverage,
        )

    def _calibration_error(
        self,
        v: Array,
        g: Array,
        c: Array,
        tau: Array,
        scores: Array,
        groups: Array,
        values: Array,
        buckets: Array,
        coverage: float = None,
        threshold: Array = None,
    ):
        prob_error, b, prob_b = self._compute_probability_error(
            v=v,
            g=g,
            tau=tau,
            scores=scores,
            groups=groups,
            values=values,
            n_buckets=len(buckets),
            return_prob_b=True,
            coverage=coverage,
            threshold=threshold,
        )
        return prob_b * prob_error, b

    def _compute_probability_error(
        self,
        v: Array,
        g: Array,
        tau: Optional[Array],
        scores: Array,
        groups: Array,
        values: Array,
        n_buckets: int,
        return_prob_b: bool = False,
        coverage: float = None,
        threshold: Array = None,
        b: Optional[Array] = None,
    ):
        prob = self._compute_probability(
            v=v,
            g=g,
            tau=tau,
            scores=scores,
            groups=groups,
            values=values,
            n_buckets=n_buckets,
            return_prob_b=return_prob_b,
            threshold=threshold,
            b=b,
        )
        if return_prob_b:
            prob, b, prob_b = prob
            return (coverage - prob) ** 2, b, prob_b
        prob, b = prob
        return (coverage - prob) ** 2, b

    def _compute_probability(
        self,
        v: Array,
        g: Array,
        tau: Optional[Array],
        scores: Array,
        groups: Array,
        values: Array,
        n_buckets: int,
        return_prob_b: bool = False,
        threshold: Array = None,
        b: Optional[Array] = None,
    ):
        if b is None:
            b = self._get_b(
                groups=groups,
                values=values,
                v=v,
                g=g,
                tau=tau,
                c=None,
                n_buckets=n_buckets,
            )
        conds = (scores <= (values if threshold is None else threshold)) * b
        prob_b = jnp.mean(b)
        prob = jnp.where(prob_b > 0, jnp.mean(conds) / prob_b, 0.0)
        if return_prob_b:
            return prob, b, prob_b
        return prob, b

    def _get_patch(
        self,
        v: Array,
        g: Array,
        c: Array,
        b: Optional[Array],
        tau: Optional[Array],
        scores: Array,
        groups: Array,
        values: Array,
        buckets: Array,
        coverage: float = None,
    ) -> Array:
        n_buckets = len(buckets)
        buckets = jnp.concatenate((-buckets[::-1][:-1], buckets))
        errors, b = vmap(
            lambda delta: self._compute_probability_error(
                v=v,
                g=g,
                tau=None,
                scores=scores,
                groups=groups,
                values=values,
                n_buckets=n_buckets,
                coverage=coverage,
                threshold=values + delta,
                b=b,
            )
        )(buckets)
        min_error = jnp.min(errors)
        indices_min_error = jnp.where(errors == min_error)[0]
        if jnp.sum(indices_min_error >= n_buckets):
            return buckets[jnp.min(indices_min_error)]
        return buckets[jnp.max(indices_min_error)]

    @staticmethod
    def _maybe_check_values(
        values: Optional[Array], test_values: Optional[Array] = None
    ):
        if jnp.any(values < 0) or jnp.any(values > 1):
            raise ValueError("All elements in `thresholds` must be within [0, 1].")

    @staticmethod
    def _check_coverage(coverage: float):
        if coverage < 0 or coverage > 1:
            raise ValueError("`coverage` must be a float between 0 and 1.")
