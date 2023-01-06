from typing import Optional, List

import jax.numpy as jnp
from jax import vmap

from fortuna.typing import Array


class SimplePredictionConformalClassifier:
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

        @vmap
        def score_fn(prob, target):
            return 1 - prob[target]

        return score_fn(val_probs, val_targets)

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
            :meth:`~fortuna.conformal.classification.simple_prediction.SimplePredictionConformalClassifier.score`.

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
            :meth:`~fortuna.conformal.classification.simple_prediction.SimplePredictionConformalClassifier.quantile`.

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
        return [jnp.where(prob > 1 - quantile)[0].tolist() for prob in test_probs]
