from fortuna.calibration.conformal.batch_mvp import BatchMVPConformalMethod
from fortuna.calibration.conformal.classification.base import ConformalClassifier

from typing import Callable, Union, List, Tuple, Optional
from fortuna.typing import Array
from fortuna.data.loader import DataLoader, InputsLoader
import jax.numpy as jnp
from jax import vmap
import numpy as np


class BatchMVPConformalClassifier(BatchMVPConformalMethod, ConformalClassifier):
    def __init__(
            self,
            score_fn: Callable[[Array, Array], Array],
            group_fns: List[Callable[[Array], Array]],
            n_classes: int,
            n_buckets: int = 100
    ):
        """
        This class implements a classification version of BatchMVP
        `[Jung et al., 2022] <https://arxiv.org/abs/2209.15145>`_,
        a multivalid conformal prediction method that satisfies coverage guarantees conditioned on group membership
        and non-conformity threshold.

        Parameters
        ----------
        score_fn: Callable[[Array, Array], Array]
            A score function mapping a batch of inputs and targets to scalar scores, one for each data point. The
            score function represents the degree of non-conformity between inputs and targets. In regression, an
            example of score function is :math:`s(x,y)=|y - h(x)|`, where `h` is an arbitrary regression model.
        group_fns: List[Callable[[Array], Array]]
            A list of group functions, each mapping input data points into boolean arrays which determine whether
            an input belongs to a certain group or not. As an example, suppose that we are interested in obtaining
            marginal coverages guarantee on both negative and positive scalar inputs.
            Then we could define groups functions
            :math:`g_1(x) = x < 0` and :math:`g_1(x) = x > 0`.
            Note that groups can be overlapping, and do not need to cover the full space of inputs.
        n_classes: int
            The number of distinct classes to classify among. The underlying assumption is that the classes are
            identified with an integer from 0 to :code:`n_classes-1`.
        n_buckets: int
            The number of buckets that defines the search space between 0 and 1 that determines the updates of the
            thresholds for the score function.
        """
        super().__init__(score_fn=score_fn, group_fns=group_fns, n_buckets=n_buckets)
        self.n_classes = n_classes

    def conformal_set(
            self,
            val_data_loader: DataLoader,
            test_inputs_loader: InputsLoader,
            error: float = 0.05,
            tol: float = 1e-4,
            n_rounds: int = 1000,
            return_max_calib_error: bool = False,
            test_thresholds: Optional[Array] = None
    ) -> Union[List[List[int]], Tuple[List[List[int]], List[Array]]]:
        """
        Compute a conformal set for each test input.

        Parameters
        ----------
        val_data_loader: DataLoader
            A data loader of validation data.
        test_inputs_loader: InputsLoader
            A loader of test input data points.
        error: float
            A desired coverage error.
        tol: float
            A tolerance for the maximum calibration error.
        n_rounds: int
            The maximum number of updates the algorithm will run for.
        return_max_calib_error: bool
            Whether to return a list of computed maximum calibration error, that is the larger calibration error
            over the different groups.
        test_thresholds: Optional[Array]
            The score thresholds computed over the test data set. These should be the output of
            `BatchMVP.threshold_score`. If provided, they will not be recomputed internally.

        Returns
        -------
        Union[List[List[int]], Tuple[List[List[int]], List[Array]]]
            The computed conformal sets for each test input. Optionally, it returns the maximum calibration errors
            computed during the algorithm.
        """
        if test_thresholds is not None and return_max_calib_error:
            raise ValueError("If `test_thresholds` is given, `return_max_calib_error` cannot be returned.")
        if test_thresholds is None:
            outs = self.threshold_score(
                val_data_loader=val_data_loader,
                test_inputs_loader=test_inputs_loader,
                error=error,
                tol=tol,
                n_rounds=n_rounds,
                return_max_calib_error=return_max_calib_error
            )
            if return_max_calib_error:
                test_thresholds, max_calib_errors = outs
            else:
                test_thresholds = outs

        c = 0
        all_ys = jnp.arange(self.n_classes)
        all_bools = []
        for inputs in test_inputs_loader:
            batch_thresholds = test_thresholds[c:c+inputs.shape[0]]
            all_bools.append(vmap(lambda y: self.score_fn(inputs, y) <= batch_thresholds, out_axes=1)(all_ys))
            c += inputs.shape[0]
        all_bools = jnp.concatenate(all_bools, axis=0)

        sizes = np.sum(all_bools, 1)
        sets = np.zeros(c, dtype=object)
        for s in np.unique(sizes):
            idx = jnp.where(sizes == s)[0]
            sets[idx] = np.nonzero(all_bools[idx])[1].reshape(len(idx), s).tolist()
        sets = sets.tolist()

        if return_max_calib_error:
            return sets, max_calib_errors
        return sets
