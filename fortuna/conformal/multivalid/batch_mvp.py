from typing import (
    Dict,
    Optional,
    Tuple,
    Union,
)

from jax import vmap
import jax.numpy as jnp

from fortuna.conformal.classification.base import ConformalClassifier
from fortuna.conformal.multivalid.base import MultivalidMethod
from fortuna.typing import Array


class BatchMVPConformalMethod(MultivalidMethod, ConformalClassifier):
    def __init__(
        self,
    ):
        """
        This class implements a classification version of BatchMVP
        `[Jung et al., 2022] <https://arxiv.org/abs/2209.15145>`_,
        a multivalid conformal prediction method that satisfies coverage guarantees conditioned on group membership
        and non-conformity threshold.
        """
        super().__init__()
        self._coverage = None

    def calibrate(
        self,
        scores: Array,
        groups: Array,
        values: Optional[Array] = None,
        test_groups: Optional[Array] = None,
        test_values: Optional[Array] = None,
        tol: float = 1e-4,
        n_buckets: int = 100,
        n_rounds: int = 1000,
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
        values: Optional[Array]
            The initial model evalutions :math:`f(x)` on the calibration data. If not provided, these are set to 0.
        test_groups: Optional[Array]
            A list of groups :math:`g(x)` computed on the test data.
            This should be a two-dimensional array of bool elements.
            The first dimension is over the data points, the second dimension is over the number of groups.
        test_values: Optional[Array]
            The initial model evaluations :math:`f(x)` on the test data. If not provided, these are set to 0.
        tol: float
            A tolerance on the reweighted average squared calibration error, i.e. :math:`\mu(g) K_2(f, g, \mathcal{D})`.
        n_buckets: int
            The number of buckets used in the algorithm. The smaller the number of buckets, the simpler the model,
            the better its generalization abilities. If not provided, We start from 2 buckets, and progressively double
            the number of buckets until we find a value for which the calibration error falls below the given
            tolerance. Such number of buckets is guaranteed to exist.
        n_rounds: int
            The maximum number of rounds to run the method for.
        coverage: float
            The desired level of coverage. This must be a scalar between 0 and 1.
        Returns
        -------
        Union[Dict, Tuple[Array, Dict]]
            A status including the number of rounds taken to reach convergence and the calibration errors computed
            during the training procedure. if `test_values` and `test_groups` are provided, the list of patches will
            be applied to `test_values`, and the calibrated test values will be returned together with the status.
        """
        if coverage < 0 or coverage > 1:
            raise ValueError("`coverage` must be a float between 0 and 1.")
        self._coverage = coverage
        return super().calibrate(
            scores=scores,
            groups=groups,
            values=values,
            test_groups=test_groups,
            test_values=test_values,
            tol=tol,
            n_buckets=n_buckets,
            n_rounds=n_rounds,
            coverage=coverage,
        )

    def calibration_error(
        self,
        scores: Array,
        groups: Array,
        values: Array,
        n_buckets: int = 10000,
        **kwargs,
    ) -> Array:
        return super().calibration_error(
            scores=scores,
            groups=groups,
            values=values,
            n_buckets=n_buckets,
            coverage=self._coverage,
        )

    def _calibration_error(
        self,
        v: Array,
        g: Array,
        scores: Array,
        groups: Array,
        values: Array,
        n_buckets: int,
        coverage: float = None,
        threshold: Array = None,
    ):
        prob_error, prob_b = self._compute_probability_error(
            v=v,
            g=g,
            scores=scores,
            groups=groups,
            values=values,
            n_buckets=n_buckets,
            return_prob_b=True,
            coverage=coverage,
            threshold=threshold,
        )
        return prob_b * prob_error

    def _compute_probability_error(
        self,
        v: Array,
        g: Array,
        scores: Array,
        groups: Array,
        values: Array,
        n_buckets: int,
        return_prob_b: bool = False,
        coverage: float = None,
        threshold: Array = None,
    ):
        prob = self._compute_probability(
            v=v,
            g=g,
            scores=scores,
            groups=groups,
            values=values,
            n_buckets=n_buckets,
            return_prob_b=return_prob_b,
            threshold=threshold,
        )
        if return_prob_b:
            prob, prob_b = prob
            return (coverage - prob) ** 2, prob_b
        return (coverage - prob) ** 2

    def _compute_probability(
        self,
        v: Array,
        g: Array,
        scores: Array,
        groups: Array,
        values: Array,
        n_buckets: int,
        return_prob_b: bool = False,
        threshold: Array = None,
    ):
        b = self._get_b(groups=groups, values=values, v=v, g=g, n_buckets=n_buckets)
        conds = (scores <= (v if threshold is None else threshold)) * b
        prob_b = jnp.mean(b)
        prob = jnp.where(prob_b > 0, jnp.mean(conds) / prob_b, 0.0)
        if return_prob_b:
            return prob, prob_b
        return prob

    def _get_patch(
        self,
        vt: Array,
        gt: Array,
        scores: Array,
        groups: Array,
        values: Array,
        buckets: Array,
        coverage: float = None,
    ) -> Array:
        return buckets[
            jnp.argmin(
                vmap(
                    lambda v: self._compute_probability_error(
                        v=vt,
                        g=gt,
                        scores=scores,
                        groups=groups,
                        values=values,
                        n_buckets=len(buckets),
                        coverage=coverage,
                        threshold=v,
                    )
                )(buckets)
            )
        ]
