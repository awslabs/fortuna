from typing import List, Optional

import jax.numpy as jnp
import numpy as np
from jax import vmap

from fortuna.typing import Array


class AdaptivePredictionConformalClassifier:
    def score(self, val_probs: Array, val_targets: Array,) -> jnp.ndarray:
        """
        Compute score function.

        Parameters
        ----------
        val_probs: Array
            A two-dimensional array of class probabilities for each validation data point.
        val_targets: Array
            A one-dimensional array of validation target variables.

        Returns
        -------
        jnp.ndarray
            The conformal scores.
        """
        if val_probs.ndim != 2:
            raise ValueError(
                """`val_probs` must be a two-dimensional array. The first dimension is over the validation
            inputs. The second is over the classes."""
            )

        perms = jnp.argsort(val_probs, axis=1)[:, ::-1]
        inv_perms = jnp.argsort(perms, axis=1)

        @vmap
        def score_fn(prob, perm, inv_perm, target):
            sorted_prob = prob[perm]
            return jnp.cumsum(sorted_prob)[inv_perm[target]]

        return score_fn(val_probs, perms, inv_perms, val_targets)

    def quantile(
        self,
        val_probs: Array,
        val_targets: Array,
        error: float = 0.05,
        scores: Optional[Array] = None,
    ) -> Array:
        """
        Compute a quantile of the scores.

        Parameters
        ----------
        val_probs: Array
            A two-dimensional array of class probabilities for each validation data point.
        val_targets: Array
            A one-dimensional array of validation target variables.
        error: float
            Coverage error. This must be a scalar between 0 and 1, extremes included.
        scores: Optional[Array]
            The conformal scores. This should be the output of
            :meth:`~fortuna.conformal.classification.adaptive_prediction.AdaptivePredictionConformalClassifier.score`.

        Returns
        -------
        float
            The conformal quantiles.
        """
        if error < 0 or error > 1:
            raise ValueError("""`error` must be a scalar between 0 and 1.""")

        if scores is None:
            scores = self.score(val_probs, val_targets)
        n = scores.shape[0]
        return jnp.quantile(scores, jnp.ceil((n + 1) * (1 - error)) / n)

    def conformal_set(
        self,
        val_probs: Array,
        test_probs: Array,
        val_targets: Array,
        error: float = 0.05,
        quantile: Optional[float] = None,
    ) -> List[List[int]]:
        """
        Coverage set of each of the test inputs, at the desired coverage error.

        Parameters
        ----------
        val_probs: Array
            A two-dimensional array of class probabilities for each validation data point.
        test_probs: Array
            A two-dimensional array of class probabilities for each test data point.
        val_targets: Array
            A one-dimensional array of validation target variables.
        error: float
            The coverage error. This must be a scalar between 0 and 1, extremes included.
        quantile: Optional[float]
            Conformal quantiles. This should be the output of
            :meth:`~fortuna.conformal.classification.adaptive_prediction.AdaptivePredictionConformalClassifier.quantile`.

        Returns
        -------
        List[List[int, ...]]
            The coverage sets.
        """
        if test_probs.ndim != 2:
            raise ValueError(
                """`test_probs` must be a two-dimensional array. The first dimension is over the validation
            inputs. The second is over the classes."""
            )

        if quantile is None:
            quantile = self.quantile(val_probs, val_targets, error)
        test_perms = jnp.argsort(test_probs, axis=1)[:, ::-1]
        test_sorted_probs = vmap(lambda prob, perm: prob[perm])(test_probs, test_perms)
        sizes = (
            (test_sorted_probs.cumsum(axis=1) > quantile).astype("int32").argmax(axis=1)
        )

        sets = np.zeros(len(sizes), dtype=object)
        for s in jnp.unique(sizes):
            idx = jnp.where(sizes == s)[0]
            sets[idx] = test_perms[idx, : s + 1].tolist()
        return sets.tolist()
