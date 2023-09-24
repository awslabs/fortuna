from typing import (
    Dict,
    Optional,
    Tuple,
    Union,
)

from jax import random
import jax.numpy as jnp

from fortuna.conformal.multivalid.iterative.multicalibrator import Multicalibrator
from fortuna.conformal.multivalid.mixins.classification.binary_multicalibrator import (
    BinaryClassificationMulticalibratorMixin,
)
from fortuna.typing import Array


class BinaryClassificationMulticalibrator(
    BinaryClassificationMulticalibratorMixin, Multicalibrator
):
    def calibrate(
        self,
        targets: Array,
        groups: Optional[Array] = None,
        probs: Optional[Array] = None,
        test_groups: Optional[Array] = None,
        test_probs: Optional[Array] = None,
        atol: float = 1e-4,
        rtol: float = 1e-6,
        min_prob_b: float = 0.1,
        n_buckets: int = 100,
        n_rounds: int = 1000,
        eta: float = 1,
        split: float = 0.8,
        bucket_types: Tuple[str, ...] = ("<=", ">="),
        **kwargs,
    ) -> Union[Dict, Tuple[Array, Dict]]:
        return super().calibrate(
            scores=targets,
            groups=groups,
            values=probs,
            test_groups=test_groups,
            test_values=test_probs,
            atol=atol,
            rtol=rtol,
            min_prob_b=min_prob_b,
            n_buckets=n_buckets,
            n_rounds=n_rounds,
            eta=eta,
            split=split,
            bucket_types=bucket_types,
            **kwargs,
        )

    def apply_patches(
        self,
        groups: Optional[Array] = None,
        probs: Optional[Array] = None,
    ) -> Array:
        return super().apply_patches(groups=groups, values=probs)

    def calibration_error(
        self,
        targets: Array,
        groups: Optional[Array] = None,
        probs: Optional[Array] = None,
        n_buckets: int = 10000,
        **kwargs,
    ) -> Array:
        return super().calibration_error(
            scores=targets,
            groups=groups,
            values=probs,
        )

    def init_probs(self, size: int) -> Array:
        """
        Initialize probabilities.

        Parameters
        ----------
        size: int
            Number of data points.

        Returns
        -------
        Array
            A probability for each data point.
        """
        return self._maybe_init_values(values=None, size=size)

    def _maybe_init_values(self, values: Optional[Array], size: Optional[int] = None):
        if values is None:
            if size is None:
                raise ValueError(
                    "If `values` is not provided, `size` must be provided."
                )
            values = 0.5 * jnp.ones(size)
            values += 0.01 * random.normal(
                random.PRNGKey(self._seed), shape=values.shape
            )

        return jnp.copy(values)
