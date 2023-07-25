from typing import (
    Callable,
    List,
    Optional,
    Tuple,
    Union,
)

import jax.numpy as jnp

from fortuna.conformal.multivalid.batch_mvp import BatchMVPConformalMethod
from fortuna.conformal.regression.base import ConformalRegressor
from fortuna.data.loader import (
    DataLoader,
    InputsLoader,
)
from fortuna.typing import Array


class Bounds:
    def __init__(self, bounds_fn: Callable[[Array, Array], Tuple[Array, Array]]):
        self.bounds_fn = bounds_fn

    def __call__(self, x: Array, t: Array) -> Tuple[Array, Array]:
        bl, br = self.bounds_fn(x, t)
        if bl.ndim > 1:
            raise ValueError(
                "Evaluations of the bounds function must e a tuple of two one-dimensional arrays. "
                f"However, the first array has shape {bl.shape}."
            )
        if br.ndim > 1:
            raise ValueError(
                "Evaluations of the bounds function must be a tuple of two one-dimensional arrays. "
                f"However, the second array has shape {bl.shape}."
            )
        if len(bl) != len(br):
            raise ValueError(
                "Evaluations of the bounds function must be a tuple of two one-dimensional arrays "
                f"with same length. However, lengths {len(bl)} and {len(br)} were found, "
                f"respectively."
            )
        return bl, br


class BatchMVPConformalRegressor(BatchMVPConformalMethod, ConformalRegressor):
    def __init__(
        self,
        score_fn: Callable[[Array, Array], Array],
        group_fns: List[Callable[[Array], Array]],
        bounds_fn: Callable[[Array, Array], Tuple[Array, Array]],
        n_buckets: int = 100,
    ):
        """
        This class implements a regression version of BatchMVP
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
        bounds_fn: Callable[[Array, Array], Array]
            A function taking a batch of input data points and respective score thresholds,
            and returning a tuple of arrays, respectively lower and upper bounds for each input.
        n_buckets: int
            The number of buckets that defines the search space between 0 and 1 that determines the updates of the
            thresholds for the score function.
        """
        super().__init__(score_fn=score_fn, group_fns=group_fns, n_buckets=n_buckets)
        self.bounds_fn = Bounds(bounds_fn=bounds_fn)

    def conformal_interval(
        self,
        val_data_loader: DataLoader,
        test_inputs_loader: InputsLoader,
        error: float = 0.05,
        tol: float = 1e-4,
        n_rounds: int = 1000,
        return_max_calib_error: bool = False,
        test_thresholds: Optional[Array] = None,
    ) -> Union[Array, Tuple[Array, List[Array]]]:
        """
        Compute a conformal interval for each test input.

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
        Union[Array, Tuple[Array, List[Array]]]
            The computed conformal intervals for each test input.
            Optionally, it returns the maximum calibration errors computed during the algorithm.
        """
        if test_thresholds is not None and return_max_calib_error:
            raise ValueError(
                "If `test_thresholds` is given, `return_max_calib_error` cannot be returned."
            )
        if test_thresholds is None:
            outs = self.threshold_score(
                val_data_loader=val_data_loader,
                test_inputs_loader=test_inputs_loader,
                error=error,
                tol=tol,
                n_rounds=n_rounds,
                return_max_calib_error=return_max_calib_error,
            )
            if return_max_calib_error:
                test_thresholds, max_calib_errors = outs
            else:
                test_thresholds = outs

        c = 0
        intervals = []
        for inputs in test_inputs_loader:
            left, right = self.bounds_fn(
                inputs, test_thresholds[c : c + inputs.shape[0]]
            )
            intervals.append(jnp.stack((left, right), axis=1))
            c += inputs.shape[0]
        intervals = jnp.concatenate(intervals, axis=0)

        if return_max_calib_error:
            return intervals, max_calib_errors
        return intervals
