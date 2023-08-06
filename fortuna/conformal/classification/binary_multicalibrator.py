from fortuna.conformal.multivalid.multicalibrator import Multicalibrator
from fortuna.typing import Array
from typing import Optional, Dict, Tuple, Union
import jax.numpy as jnp


class BinaryClassificationMulticalibrator(Multicalibrator):
    def calibrate(
        self,
        targets: Array,
        groups: Array,
        probs: Optional[Array] = None,
        test_groups: Optional[Array] = None,
        test_probs: Optional[Array] = None,
        tol: float = 1e-4,
        n_buckets: int = 100,
        n_rounds: int = 1000,
        **kwargs,
    ) -> Union[Dict, Tuple[Array, Dict]]:
        self._check_targets(targets)
        self._check_probs(probs)

        return super().calibrate(
            scores=targets,
            groups=groups,
            values=probs,
            test_groups=test_groups,
            test_values=test_probs,
            tol=tol,
            n_buckets=n_buckets,
            n_rounds=n_rounds,
            **kwargs
        )

    def calibration_error(
        self,
        targets: Array,
        groups: Array,
        probs: Array,
        n_buckets: int = 10000,
        **kwargs,
    ) -> Array:
        self._check_targets(targets)
        self._check_probs(probs)

        return super().calibration_error(
            scores=targets,
            groups=groups,
            values=probs,
        )

    def apply_patches(
        self,
        groups: Array,
        probs: Optional[Array] = None,
    ) -> Array:
        self._check_probs(probs)

        return super().apply_patches(
            groups=groups,
            values=probs
        )

    @staticmethod
    def _check_targets(targets: Array):
        if targets.ndim != 1:
            raise ValueError("`targets` must be a 1-dimensional array of integers.")
        if set(jnp.unique(targets)) != {0, 1}:
            raise ValueError("All values in `targets` must be 0 or 1.")
        if targets.dtype not in ["int32", "int64"]:
            raise ValueError("All elements in `targets` must be integers")

    @staticmethod
    def _check_probs(probs: Array):
        if probs.ndim != 1:
            raise ValueError("`probs` must be a 1-dimensional array representing the probability that the "
                             "target variable is 1.")
        if jnp.any(probs > 1) or jnp.any(probs < 0):
            raise ValueError("All values in `probs` must be 0 or 1.")
