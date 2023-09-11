from typing import List

import jax.numpy as jnp
import numpy as np

from fortuna.conformal.classification.base import ConformalClassifier
from fortuna.conformal.multivalid.batch_mvp import BatchMVPConformalMethod
from fortuna.typing import Array


class BatchMVPConformalClassifier(BatchMVPConformalMethod, ConformalClassifier):
    def __init__(
        self,
        seed: int = 0
    ):
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

    def conformal_set(
        self,
        class_scores: Array,
        thresholds: Array,
    ) -> List[List[int]]:
        """
        Compute a conformal set for each input.

        Parameters
        ----------
        class_scores: Array
            A two-dimensional array of scores. The first dimension is over the different inputs.
            The second dimension is over all the possible classes. For example, if there are 10 classes,
            the first row of `class_scores` show be :math:`[s(x_1, 0), \dots, s(x_1, 9)]`.
        thresholds: Array
            A one-dimensional array of thresholds over the different inputs. This should be obtained from the `calibrate`
            method.

        Returns
        -------
        List[List[int]]
            Conformal sets for each input data point.
        """
        if class_scores.ndim != 2:
            raise ValueError(
                "`class_scores` must bse a 2-dimensional array. "
                "The first dimension is over the different inputs. "
                "The second dimension is over all the possible classes."
            )
        if thresholds.ndim != 1:
            raise ValueError("`thresholds` must be a 1-dimensional array.")
        if class_scores.shape[0] != thresholds.shape[0]:
            raise ValueError(
                "The first dimension of `class_scores` and `thresholds` must be over the same input data "
                "points."
            )
        bools = class_scores <= thresholds[:, None]

        sizes = np.sum(bools, 1)
        sets = np.zeros(bools.shape[0], dtype=object)
        for s in np.unique(sizes):
            idx = jnp.where(sizes == s)[0]
            sets[idx] = np.nonzero(bools[idx])[1].reshape(len(idx), s).tolist()
        return sets.tolist()
