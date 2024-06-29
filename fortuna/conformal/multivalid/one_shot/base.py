import abc
import logging
from typing import (
    Optional,
    Union,
)

from jax import vmap
import jax.numpy as jnp

from fortuna.conformal.multivalid.base import MultivalidMethod
from fortuna.typing import Array


class OneShotMultivalidMethod(MultivalidMethod):
    def __init__(self, seed: int = 0):
        """
        A base one-shot multivalid method.

        Parameters
        ----------
        seed: int
            Random seed.
        """
        super().__init__(seed=seed)
        self._patches = None
        self._n_buckets = None

    def calibrate(
        self,
        scores: Array,
        values: Array = None,
        test_values: Optional[Array] = None,
        n_buckets: int = 100,
        min_prob_b: Union[float, str] = "auto",
    ) -> Union[None, Array]:
        """
        Calibrate the model by finding a list of patches to the model that bring the calibration error below a
        certain threshold.

        Parameters
        ----------
        scores: Array
            A list of scores :math:`s(x, y)` computed on the calibration data.
            This should be a one-dimensional array of elements between 0 and 1.
        values: Optional[Array]
            The initial model evalutions :math:`f(x)` on the calibration data. If not provided, these are set to 0.
        test_values: Optional[Array]
            The initial model evaluations :math:`f(x)` on the test data. If not provided, these are set to 0.
        n_buckets: int
            The number of buckets used in the algorithm.
        min_prob_b: Union[float, str]
            Minimum probability of the conditioning set :math:`B_t` for the patch to be applied.
            If "auto", it will be chosen based on the number of buckets and dimension of the scores.

        Returns
        -------
        Union[Dict, Tuple[Array, Dict]]
            A status including the number of rounds taken to reach convergence and the calibration errors computed
            during the training procedure. if `test_values` is provided, the list of patches will
            be applied, and the calibrated test values will be returned together with the status.
        """
        if test_values is not None and values is None:
            raise ValueError(
                "If `test_values is provided, `values` must also be provided."
            )
        if min_prob_b != "auto" and (min_prob_b < 0 or min_prob_b > 1):
            raise ValueError(
                "`min_prob_b` must be greater than or equal to 0 and less than or equal to 1."
            )
        if values is None:
            raise ValueError("`values` must be provided.")

        self._check_scores(scores)
        scores = self._process_scores(scores)
        n_dims = scores.shape[1]

        min_prob_b = self._maybe_init_min_prob_b(
            min_prob_b=min_prob_b, n_buckets=n_buckets, n_dims=n_dims
        )

        self._maybe_check_values(values, test_values)
        values, test_values = self._maybe_process_values(values, test_values)

        self.n_buckets = n_buckets
        buckets = self._get_buckets(n_buckets)

        self._patches = vmap(
            lambda v: vmap(
                lambda c: self._get_patch(
                    v=v, c=c, scores=scores[:, c], values=values, min_prob_b=min_prob_b
                )
            )(jnp.arange(n_dims))
        )(buckets)

        if test_values is not None:
            return self.apply_patches(test_values)

    def apply_patches(
        self,
        values: Array,
    ) -> Array:
        if self._patches is None:
            logging.warning("No patches available.")
            return values

        buckets = self._get_buckets(self.n_buckets)
        unique_values = jnp.unique(values)

        n_dims = 1 if values.ndim == 1 else values.shape[1]

        b = vmap(
            lambda v: vmap(
                lambda c: self._get_b(values, v, c, self.n_buckets), out_axes=1
            )(jnp.arange(n_dims)),
            out_axes=1,
        )(unique_values)

        return self._patch(
            values=values, b=b, unique_values=unique_values, buckets=buckets
        )

    @staticmethod
    def _get_b(
        values: Array,
        v: Array,
        c: Optional[Array],
        n_buckets: int,
    ) -> Array:
        return jnp.abs(values - v) < 0.5 / n_buckets

    def _patch(
        self, values: Array, b: Array, unique_values: Array, buckets: Array
    ) -> Array:
        patched_values = jnp.copy(values)
        for i, v in enumerate(unique_values):
            for c in range(b.shape[2]):
                idx_v = jnp.argmin(jnp.abs(buckets - v))
                if values.ndim == 1:
                    patched_values = patched_values.at[b[:, i, c]].set(
                        self._patches[idx_v, c]
                    )
                else:
                    patched_values = patched_values.at[b[:, i, c], c].set(
                        self._patches[idx_v, c]
                    )
        return patched_values

    @abc.abstractmethod
    def _get_patch(
        self, v: Array, c: Array, scores: Array, values: Array, min_prob_b: float
    ) -> Array:
        pass
