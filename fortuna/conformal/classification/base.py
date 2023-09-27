import abc
from typing import (
    List,
    Tuple,
)

from jax import vmap
import jax.numpy as jnp
import numpy as np

from fortuna.typing import Array


class ConformalClassifier:
    """
    A base conformal classifier class.
    """

    def is_in(self, values: Array, conformal_sets: List) -> Array:
        """
        Check whether the values lie within their respective conformal sets.

        Parameters
        ----------
        values: Array
            Values to check if they lie in the respective conformal sets.
        conformal_sets: Array
            A conformal set for each input data point.

        Returns
        -------
        Array
            An array of ones or zero, indicating whether the values lie within their respective conformal sets.
        """
        return jnp.array([v in s for v, s in zip(values.tolist(), conformal_sets)])

    @abc.abstractmethod
    def score_fn(
        self,
        probs: Array,
        targets: Array,
    ):
        pass

    @staticmethod
    def _get_conformal_sets_from_scores(
        val_scores: Array,
        test_scores: Array,
        error: float,
    ) -> List[List[int]]:
        conds = jnp.sum(val_scores[:, None, None] > test_scores[None], axis=0) < (
            1 - error
        ) * (len(val_scores) + 1)
        sizes = conds.sum(1)

        sets = np.zeros(len(test_scores), dtype=object)
        for us in jnp.unique(sizes):
            idx = jnp.where(sizes == us)[0]
            if us == 0:
                sets[idx] = [len(idx) * []]
            else:
                sets[idx] = np.where(conds[idx])[1].reshape(-1, us).tolist()

        return sets.tolist()

    @abc.abstractmethod
    def get_scores(self, *args, **kwargs) -> Tuple[Array, Array]:
        pass


class SplitConformalClassifier(ConformalClassifier, abc.ABC):
    def get_scores(
        self, val_probs: Array, val_targets: Array, test_probs: Array
    ) -> Tuple[Array, Array]:
        val_scores = self.score_fn(val_probs, val_targets)
        test_scores = vmap(
            lambda i: self.score_fn(
                test_probs, i * jnp.ones(len(test_probs), dtype="int32")
            ),
            out_axes=1,
        )(jnp.arange(val_probs.shape[1]))
        return val_scores, test_scores

    def conformal_set(
        self, val_probs: Array, val_targets: Array, test_probs: Array, error: float
    ) -> List[List[int]]:
        val_scores, test_scores = self.get_scores(
            val_probs=val_probs, val_targets=val_targets, test_probs=test_probs
        )
        return super()._get_conformal_sets_from_scores(
            val_scores=val_scores, test_scores=test_scores, error=error
        )


class CVPlusConformalClassifier(ConformalClassifier):
    def conformal_set(
        self,
        cross_val_probs: List[Array],
        cross_val_targets: List[Array],
        cross_test_probs: List[Array],
        error: float,
    ) -> List[List[int]]:
        val_scores, test_scores = self.get_scores(
            cross_val_probs=cross_val_probs,
            cross_val_targets=cross_val_targets,
            cross_test_probs=cross_test_probs,
        )
        return super()._get_conformal_sets_from_scores(
            val_scores=val_scores, test_scores=test_scores, error=error
        )

    def get_scores(
        self,
        cross_val_probs: List[Array],
        cross_val_targets: List[Array],
        cross_test_probs: List[Array],
    ) -> Tuple[Array, Array]:
        val_scores, test_scores = [], []
        for val_probs, val_targets, test_probs in zip(
            cross_val_probs, cross_val_targets, cross_test_probs
        ):
            val_scores.append(self.score_fn(val_probs, val_targets))
            test_scores.append(
                vmap(
                    lambda i: self.score_fn(
                        test_probs, i * jnp.ones(len(test_probs), dtype="int32")
                    ),
                    out_axes=1,
                )(jnp.arange(cross_val_probs[0].shape[1]))
            )

        return jnp.concatenate(val_scores), jnp.concatenate(test_scores, axis=0)
