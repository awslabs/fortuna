import abc
import logging
from typing import (
    Dict,
    Optional,
    Tuple,
    Union,
)

from jax import (
    random,
    vmap,
)
import jax.numpy as jnp

from fortuna.conformal.multivalid.base import MultivalidMethod
from fortuna.typing import Array


class MaxCoverageFixedPrecisionBinaryClassificationCalibrator:
    def __init__(self, seed: int = 0):
        """
        A base iterative multivalid method.

        Parameters
        ----------
        seed: int
            Random seed.
        """
        super().__init__(seed=seed)
        self._patches = []
        self._eta = None

    def calibrate(
        self,
        targets: Array,
        probs: Array,
        test_probs: Optional[Array] = None,
        n_buckets: int = 100,
        **kwargs,
    ) -> Union[Dict, Tuple[Array, Dict]]:
        """
        Calibrate the model by finding a list of patches to the model that bring the calibration error below a
        certain threshold.

        Parameters
        ----------
        targets: Array
            Binary target variables.
        probs: Optional[Array]
            Probability that the target is 1 for each input of the calibration set.
        test_probs: Optional[Array]
            Probability that the target is 1 for each input of the test set.
        n_buckets: int
            The number of buckets used in the algorithm.

        Returns
        -------
        Union[Dict, Tuple[Array, Dict]]
            A status including the number of rounds taken to reach convergence and the calibration errors computed
            during the training procedure. if `test_values` and `test_groups` are provided, the list of patches will
            be applied to `test_values`, and the calibrated test values will be returned together with the status.
        """


    def apply_patches(
        self,
        groups: Optional[Array] = None,
        values: Optional[Array] = None,
    ) -> Array:
        """
        Apply the patches to the model evaluations.

        Parameters
        ----------
        groups: Array
            A list of groups :math:`g(x)` evaluated over some inputs.
            This should be a two-dimensional array of bool elements.
            The first dimension is over the data points, the second dimension is over the number of groups.
        values: Optional[Array]
            The initial model evaluations :math:`f(x)` evaluated over some inputs. If not provided, these are set to 0.

        Returns
        -------
        Array
            The calibrated values.
        """
        if groups is None and values is None:
            raise ValueError(
                "At least one between `groups` and `values` must be provided."
            )
        if not len(self._patches):
            logging.warning("No patches available.")
            return values

        values = self._maybe_init_values(
            values, groups.shape[0] if groups is not None else None
        )
        self._maybe_check_values(values)

        groups = self._init_groups(groups, values.shape[0])
        self._maybe_check_groups(groups)

        buckets = self._get_buckets(n_buckets=self.n_buckets)
        values = vmap(lambda v: self._round_to_buckets(v, buckets))(values)

        for gt, vt, ct, patch in self._patches:
            bt = self._get_b(
                groups=groups, values=values, v=vt, g=gt, c=ct, n_buckets=self.n_buckets
            )
            values = self._patch(values=values, patch=patch, bt=bt, ct=ct, eta=self.eta)
        return values

    def objective_fn(self, probs: Array, targets: Array, positive_threshold: float, negative_threshold: float):
        bs = {
            "low": probs <= negative_threshold,
            "midlow": (probs > negative_threshold) * (probs < 0.5),
            "midhigh": (probs >= 0.5) * (probs < positive_threshold),
            "high": probs >= positive_threshold
        }
        buckets = jnp.linspace(0, 1, self.n_buckets)
        vmap(lambda b: vmap(lambda s: jnp.mean((probs[b] + s) * targets[b]) / jnp.mean(b))(buckets)

        def _get_precisions(b: Array, s: Array):
            probs[b]


    def _get_precisions(self, bs: Dict, probs: Array, targets: Array, positive_threshold: float, negative_threshold: float):
            bs[""]

