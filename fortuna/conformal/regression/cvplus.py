import jax.numpy as jnp
from fortuna.typing import Array
from typing import List


class CVPlusConformalRegressor:
    """
    This class implements the CV+ method introduced in
    `Barber et al., 2021 <https://www.stat.cmu.edu/~ryantibs/papers/jackknife.pdf>`__. It is an extension of the
    jackknife+ method, introduced in the same work, that consider a K-Fold instead of a leave-one-out strategy. If
    :code:`K=n`, where :code:`n` is the total number of training data, then CV+ reduces to jackknife+.
    """
    def conformal_interval(
            self,
            cross_val_outputs: List[Array],
            cross_val_targets: List[Array],
            cross_test_outputs: List[Array],
            error: float
    ) -> jnp.ndarray:
        """
        Coverage interval of each of the test inputs, at the desired coverage error. This is supported only for
        one-dimensional target variables.

        Parameters
        ----------
        cross_val_outputs: List[Array]
            Outputs of the models used during cross validation evaluated at their respective validation inputs. More
            precisely, we assume the training data has been jointly partitioned in :code:`K` subsets. The i-th element
            of the list of :code: `cross_val_outputs` is a model trained on all data but the i-th partition, and has
            been evaluated at the inputs of the partition i-th itself, for :code:`i=1, 2, ..., K`.
        cross_val_targets: List[Array]
            Target variables organized in the same partitions used for `cross_val_outputs`. More precisely, the i-th
            element of :code:`cross_val_targets` includes the array of target variables of the i-th partition of the
            training data, for :code:`i=1, 2, ..., K`.
        cross_test_outputs: List[Array]
            Outputs of the models used during cross validation evaluated at the test inputs. More precisely, consider
            the same partition of data as the one used for :code:`cross_val_outputs`. Then the i-th element of
            :code:`cross_test_outputs` represents the outputs of the model that has been trained upon all the training
            data but the i-th partition, and evaluated at the test inputs, for :code:`i=1, 2, ..., K`.
        error: float
            The desired coverage error. This must be a scalar between 0 and 1, extremes included.
        Returns
        -------
        jnp.ndarray
            The conformal intervals. The two components of the second axis correspond to the left and right interval
            bounds.
        """
        if type(cross_val_outputs) != list:
            raise TypeError("`cross_val_outputs` must be a list of arrays.")
        if type(cross_val_targets) != list:
            raise TypeError("`cross_val_targets` must be a list of arrays.")
        if type(cross_test_outputs) != list:
            raise TypeError("`cross_test_outputs` must be a list of arrays.")
        for i, (mu, y, mu_test) in enumerate(zip(cross_val_outputs, cross_val_targets, cross_test_outputs)):
            if mu.shape[0] != y.shape[0]:
                raise ValueError("The first dimension of the i-th element in `cross_val_outputs` must be the same as "
                                 "the one of the i-th element in `cross_val_targets`.")
            if mu.ndim == 1:
                cross_val_outputs[i] = mu[:, None]
            elif mu.shape[1] != 1:
                raise ValueError(
                    "This method is supported only for scalar model outputs only. However, an element of "
                    "`cross_val_outputs` has second dimension greater than 1.")
            if y.ndim == 1:
                cross_val_targets[i] = y[:, None]
            elif y.shape[1] != 1:
                raise ValueError("This method is supported only for scalar target variables. However, an element of "
                                 "`cross_val_targets` has second dimension greater than 1.")
            if mu_test.ndim == 1:
                cross_test_outputs[i] = mu_test[:, None]
            elif mu_test.shape[1] != 1:
                raise ValueError("This method is supported only for scalar model outputs only. However, an element of "
                                 "`cross_test_outputs` has second dimension greater than 1.")

        r = [jnp.abs(y - mu) for y, mu in zip(cross_val_targets, cross_val_outputs)]
        left = jnp.concatenate([mu[None] - ri[:, None] for mu, ri in zip(cross_test_outputs, r)], 0)
        right = jnp.concatenate([mu[None] + ri[:, None] for mu, ri in zip(cross_test_outputs, r)])

        qleft = jnp.quantile(left, q=error, axis=0)
        qright = jnp.quantile(right, q=1 - error, axis=0)
        return jnp.array(list(zip(qleft, qright))).squeeze(2)
