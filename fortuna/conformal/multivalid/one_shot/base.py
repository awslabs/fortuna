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
import numpy as np

from fortuna.typing import Array


class MultivalidMethod:
    def __init__(self, seed: int = 0):
        """
        A base one-shot multivalid method.

        Parameters
        ----------
        seed: int
            Random seed.
        """
        self._patches = []
        self._n_buckets = None
        self._seed = seed

    def calibrate(
        self,
        scores: Array,
        values: Optional[Array] = None,
        test_values: Optional[Array] = None,
        n_buckets: int = 100,
    ) -> Union[Dict, Tuple[Array, Dict]]:
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

        self._check_scores(scores)
        scores = self._process_scores(scores)
        n_dims = scores.shape[1]

        values = self._maybe_init_values(values)
        self._maybe_check_values(values, test_values)

        self.n_buckets = n_buckets
        buckets = self._get_buckets(n_buckets)

        self._patches = vmap(
            lambda v: vmap(
                lambda c: self._patch(
                    v=v,
                    c=c,
                    scores=scores[:, c],
                    values=values,
                    n_buckets=n_buckets,
                    **kwargs,
                )
            )(jnp.arange(n_dims))
        )(buckets)