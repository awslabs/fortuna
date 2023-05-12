from typing import (
    Callable,
    Tuple,
    Union,
)

import jax.numpy as jnp
import numpy as np

from fortuna.conformal.regression.base import ConformalRegressor
from fortuna.typing import Array


class EnbPI(ConformalRegressor):
    def __init__(
        self, aggregation_fun: Callable[[Array], Array] = lambda x: jnp.mean(x, 0)
    ):
        """
        Ensemble Batch Prediction Intervals (EnbPI) is a conformal prediction algorithm for time series regression.
        By bootstrapping the data and training the model on each of the bootstrap samples, EnbPI is able to compute
        conformal intervals that satisfy an approximate marginal guarantee on each test data point. Furthermore,
        EnbPI can incorporate online feedback from incoming batches of data, and improve the conformal intervals
        without having to retrain the model.

        Parameters
        ----------
        aggregation_fun: Callable[[Array], Array]
            Aggregation function that takes a group of model predictions from different bootstrap samples and reduces
            them to a single one.
        """
        self.aggregation_fun = aggregation_fun

    def conformal_interval(
        self,
        bootstrap_indices: Array,
        bootstrap_train_preds: Array,
        bootstrap_test_preds: Array,
        train_targets: Array,
        error: float,
        return_residuals: bool = False,
    ) -> Union[Array, Tuple[Array, Array]]:
        """
        Compute a coverage interval for each of the test inputs of the time series, at the desired coverage error.
        This is supported only for one-dimensional target variables, and for one times series at the time.

        Parameters
        ----------
        bootstrap_indices: Array
            The indices, randomly sampled with replacement, of the training data in the time series used to train the
            model. The first dimension is over the different samples of indices. The second dimension contain the
            data points for each sample, which must be as many as thnumber of data points in the time series used for
            training. A simple way of obtaining indices randomly sampled with replacement is
            :code:`numpy.random.choice(T, size=(B, T))`,
            where :code:`T` is the number of training points in the time series,
            and :code:`B` is the number of bootstrap samples.
            It is the user job to make sure that the models are trained upon the data corresponding to the random
            indices.
        bootstrap_train_preds: Array
            Model predictions for each of the bootstrap samples of data of the time series used for training the model,
            evaluated at each of the training data inputs of the time series. The first dimension is over the
            different bootstrap samples. The second dimensions is over the training inputs.
            There may be a third dimension, corresponding to the dimensionality of the predictions,
            but if so this must be one.
        bootstrap_test_preds: Array
            Model predictions for each of the bootstrap samples of data of the time series used for training the model,
            evaluated at each of the test data inputs of the time series. The first dimension is over the
            different bootstrap samples. The second dimensions is over the test inputs. There may be a third dimension,
            corresponding to the dimensionality of the predictions, but if so this must be one.
        train_targets: Array
            The target variables of the training data points in the time series.
        error: float
            The desired coverage error. This must be a scalar between 0 and 1, extremes included.
        return_residuals: bool
            If True, return the residual errors computed over the training data. These are used in
            :meth:`~fortuna.conformal.regression.enbpi.EnbPI.conformal_interval_from_residuals`.

        Returns
        -------
        Union[Array, Tuple[Array, Array]]
            The conformal intervals. The two components of the second dimension correspond to the left and right
            interval bounds. If :code:`return_residuals` is set to True, then it returns also the residuals computed on
            the training set.
        """
        n_bootstraps, n_train_times = bootstrap_indices.shape
        if (
            jnp.min(bootstrap_indices) < 0
            or jnp.max(bootstrap_indices) >= n_train_times
        ):
            raise ValueError(
                f"All elements f `bootstrap_indices` must be integers from 0 to {n_train_times - 1} "
                f"corresponding to the indices of the data used for training in each of the bootstrap "
                f"samples."
            )
        if bootstrap_train_preds.shape[0] != n_bootstraps:
            raise ValueError(
                "The first dimension of `bootstrap_train_preds` and `bootstrap_indices` corresponds to "
                "the number of bootstrap samples, and must have the same size. However, "
                f"{bootstrap_train_preds.shape[0]} and {n_bootstraps} were found, respectively."
            )
        if bootstrap_train_preds.shape[1] != n_train_times:
            raise ValueError(
                "The second dimension of `bootstrap_train_preds` and `bootstrap_indices` corresponds to "
                "the number of data points in the time series used for training, "
                "and must have the same size. "
                f"However, {bootstrap_train_preds.shape[1]} and {n_train_times} were found, respectively."
            )
        if bootstrap_test_preds.shape[0] != n_bootstraps:
            raise ValueError(
                "The first dimension of `bootstrap_test_preds` and `bootstrap_indices` corresponds to the "
                "number of bootstrap samples, and must have the same size. However, "
                f"{bootstrap_test_preds.shape[0]} and {n_bootstraps} were found, respectively."
            )
        if bootstrap_train_preds.ndim == 3:
            if bootstrap_train_preds.shape[2] == 1:
                bootstrap_train_preds = bootstrap_train_preds.squeeze(2)
            else:
                raise ValueError(
                    "This method is supported only for scalar model predictions. However, `bootstrap_train_preds` has "
                    "third dimension greater than 1."
                )
        if bootstrap_test_preds.ndim == 3:
            if bootstrap_test_preds.shape[2] == 1:
                bootstrap_test_preds = bootstrap_test_preds.squeeze(2)
            else:
                raise ValueError(
                    "This method is supported only for scalar model predictions. However, `bootstrap_test_preds` has "
                    "third dimension greater than 1."
                )
        if train_targets.shape[0] != bootstrap_train_preds.shape[1]:
            raise ValueError(
                "The first dimension of `train_targets` and the second dimension of "
                "`bootstrap_train_preds` correspond to the number of data points in the time series used "
                "for training, and must have the same size. However, "
                f"{train_targets.shape[0]} and {bootstrap_train_preds.shape[1]} were found, respectively."
            )
        if train_targets.ndim == 2:
            if train_targets.shape[1] == 1:
                train_targets = train_targets.squeeze(1)
            else:
                raise ValueError(
                    "This method is supported only for scalar target variables. However, `train_targets` has "
                    "second dimension greater than 1."
                )

        in_bootstrap_indices = np.zeros((n_bootstraps, n_train_times), dtype=bool)
        np.put_along_axis(in_bootstrap_indices, bootstrap_indices, values=1, axis=1)
        aggr_bootstrap_test_preds = np.zeros(
            (n_train_times,) + bootstrap_test_preds.shape[1:]
        )
        train_residuals = np.zeros((n_train_times,) + train_targets.shape[1:])

        for t in range(n_train_times):
            which_bootstraps = np.where(~(in_bootstrap_indices[:, t]))[0]
            if len(which_bootstraps) > 0:
                aggr_bootstrap_train_pred = self.aggregation_fun(
                    bootstrap_train_preds[which_bootstraps, t]
                )
                train_residuals[t] = np.abs(
                    train_targets[t] - aggr_bootstrap_train_pred
                )
                aggr_bootstrap_test_preds[t] = self.aggregation_fun(
                    bootstrap_test_preds[which_bootstraps]
                )
            else:
                train_residuals[t] = np.abs(train_targets[t])

        test_quantiles = jnp.quantile(aggr_bootstrap_test_preds, q=1 - error, axis=0)
        residuals_quantile = jnp.quantile(train_residuals, q=1 - error, axis=0)

        left = test_quantiles - residuals_quantile
        right = test_quantiles + residuals_quantile

        conformal_intervals = jnp.array(list(zip(left, right)))
        if not return_residuals:
            return conformal_intervals
        return conformal_intervals, train_residuals

    def conformal_interval_from_residuals(
        self,
        train_residuals: Array,
        bootstrap_new_train_preds: Array,
        bootstrap_new_test_preds: Array,
        new_train_targets: Array,
        error: float,
    ) -> Union[Array, Tuple[Array, Array]]:
        """
        Compute a coverage interval for each of the test inputs of the time series, at the desired coverage error.
        This is supported only for one-dimensional target variables, and for one times series at the time.
        This method assumes the residuals over the training sets have already been computed via
        :meth:`~fortuna.conformal.regression.enbpi.EnbPI.conformal_interval`
        using the flag :code:`return_residuals=True`.
        If so, this method takes in the predictions for each bootstrap samples over the new incoming training and test
        data points, and computes conformal intervals without having to retrain the model. Compared to predicting
        conformal intervals over all test data points at once with
        :meth:`~fortuna.conformal.regression.enbpi.EnbPI.conformal_interval`,
        this method exploits incoming data points as a form of online feedback,
        and improves the conformal intervals accordingly.

        Parameters
        ----------
        train_residuals: Array
            Residuals over the training data points computed via
            :meth:`~fortuna.conformal.regression.enbpi.EnbPI.conformal_interval` using the flag
            :code:`return_residuals=True`.
        bootstrap_new_train_preds: Array
            Model predictions for each of the bootstrap samples of data of the time series used for training the model,
            evaluated at the new incoming training data inputs of the time series. The first dimension is over the
            different bootstrap samples. The second dimensions is over the new training inputs.
            There may be a third dimension, corresponding to the dimensionality of the predictions,
            but if so this must be one. As an example, suppose that the bootstrap has been done over data points from
            1 to T, and the model was trained over each bootstrap sample. A prediction was then made on a batch of
            data points from T+1 to T+T1. After the data from T+1 to T+T1 is observed, this will be used as the new
            batch of training data.
        bootstrap_new_test_preds: Array
            Model predictions for each of the bootstrap samples of data of the time series used for training the model,
            evaluated at the new incoming test data inputs of the time series. The first dimension is over the
            different bootstrap samples. The second dimensions is over the new test inputs.
            There may be a third dimension, corresponding to the dimensionality of the predictions,
            but if so this must be one. As an example, suppose that the bootstrap has been done over data points from
            1 to T, and the model was trained over each bootstrap sample. A prediction was then made on a batch of
            data points from T+1 to T+T1. After the data from T+1 to T+T1 is observed, data from T+T1+1 to T+2*T1 may
            be taken as the new batch of test data.
        new_train_targets: Array
            The target variables of the new batch of training data points in the time series.
        error: float
            The desired coverage error. This must be a scalar between 0 and 1, extremes included.
        Returns
        -------
        Union[Array, Tuple[Array, Array]]
            - The conformal intervals. The two components of the second dimension correspond to the left and right
            interval bounds.
            - A new set of residuals, which includes the residuals computed on the new batch of training data points.
            The number of stored training residuals is kept constant by eliminating the oldest ones.
        """
        if bootstrap_new_train_preds.shape[0] != bootstrap_new_test_preds.shape[0]:
            raise ValueError(
                "The first dimensions of `bootstrap_new_train_preds` and `bootstrap_new_test_preds` "
                "correspond to the number of bootstrap samples, "
                "and must have the same size. However, "
                f"{bootstrap_new_train_preds.shape[0]} and "
                f"{bootstrap_new_test_preds.shape[0]} were found, respectively."
            )
        if bootstrap_new_train_preds.ndim == 3:
            if bootstrap_new_train_preds.shape[2] == 1:
                bootstrap_new_train_preds = bootstrap_new_train_preds.squeeze(2)
            else:
                raise ValueError(
                    "This method is supported only for scalar model predictions. However, "
                    "`bootstrap_new_train_preds` has third dimension greater than 1."
                )
        if bootstrap_new_test_preds.ndim == 3:
            if bootstrap_new_test_preds.shape[2] == 1:
                bootstrap_new_test_preds = bootstrap_new_test_preds.squeeze(2)
            else:
                raise ValueError(
                    "This method is supported only for scalar model predictions. "
                    "However, `bootstrap_new_test_preds` has third dimension greater than 1."
                )
        if new_train_targets.shape[0] != bootstrap_new_train_preds.shape[1]:
            raise ValueError(
                "The first dimension of `new_train_targets` and the second dimension of "
                "`bootstrap_new_train_preds` correspond to the number of data points in the time series "
                "used for training, and must have the same size. However, "
                f"{new_train_targets.shape[0]} and {bootstrap_new_train_preds.shape[1]} were found, "
                f"respectively."
            )
        if new_train_targets.ndim == 2:
            if new_train_targets.shape[1] == 1:
                new_train_targets = new_train_targets.squeeze(1)
            else:
                raise ValueError(
                    "This method is supported only for scalar target variables. However, `new_train_targets` has "
                    "second dimension greater than 1."
                )

        train_residuals[:-1] = train_residuals[1:]
        train_residuals[-1] = np.abs(
            new_train_targets - self.aggregation_fun(bootstrap_new_train_preds)
        )
        aggr_test_preds = self.aggregation_fun(bootstrap_new_test_preds)
        residuals_quantile = jnp.quantile(train_residuals, q=1 - error, axis=0)

        left = aggr_test_preds - residuals_quantile
        right = aggr_test_preds + residuals_quantile

        return jnp.array(list(zip(left, right))), train_residuals
