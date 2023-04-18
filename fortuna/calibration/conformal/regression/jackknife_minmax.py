import jax.numpy as jnp

from fortuna.typing import Array
from fortuna.calibration.conformal.regression.base import ConformalRegressor


class JackknifeMinmaxConformalRegressor(ConformalRegressor):
    """
    This class implements the jackknife-minmax method introduced in
    `Barber et al., 2021 <https://www.stat.cmu.edu/~ryantibs/papers/jackknife.pdf>`__. Jackknife-minmax guarantees the
    desired minimal coverage :math:`1-α`, while jackknife+ only guarantees a minimal coverage of
    :math:`1 - 2α`.
    """

    def conformal_interval(
        self,
        loo_val_outputs: Array,
        loo_val_targets: Array,
        loo_test_outputs: Array,
        error: float,
    ) -> jnp.ndarray:
        """
        Coverage interval of each of the test inputs, at the desired coverage error. This is supported only for
        one-dimensional target variables.

        Parameters
        ----------
        loo_val_outputs: Array
            Outputs of the models used during leave-out-out evaluated at their respective left-out validation inputs.
            More precisely, the i-th element of :code: `loo_val_outputs` is a model that has been trained upon all the
            training data but the i-th data point, and evaluated at the i-th input, for all training data points.
        loo_val_targets: Array
            The array of target variables of the left-out data points in the leave-one-out procedure described for
            `loo_val_outputs`.
        loo_test_outputs: Array
            Outputs of the models used during leave-one-out evaluated at the test inputs. More precisely, consider
            the same leave-one-out procedure as the one used for :code:`loo_val_outputs`. Then the i-th element of
            the first dimension of :code:`loo_test_outputs` represents the outputs of the model that has been trained
            upon all the training data but the i-th data point, and evaluated at the test inputs. The second dimension
            of :code:`loo_test_outputs` is over the different test inputs.
        error: float
            The desired coverage error. This must be a scalar between 0 and 1, extremes included.
        Returns
        -------
        jnp.ndarray
            The conformal intervals. The two components of the second axis correspond to the left and right interval
            bounds.
        """
        if loo_val_outputs.shape[0] != loo_val_targets.shape[0]:
            raise ValueError(
                "The first dimension of `loo_val_outputs` and `loo_val_targets` must coincide. However, "
                f"{loo_val_outputs.shape[0]} and {loo_val_targets.shape[0]} were found, respectively."
            )
        if loo_val_outputs.ndim == 1:
            loo_val_outputs = loo_val_outputs[:, None]
        elif loo_val_outputs.shape[1] != 1:
            raise ValueError(
                "This method is supported only for scalar model outputs only. However, `loo_val_outputs` has second "
                "dimension greater than 1."
            )
        if loo_val_targets.ndim == 1:
            loo_val_targets = loo_val_targets[:, None]
        elif loo_val_targets.shape[1] != 1:
            raise ValueError(
                "This method is supported only for scalar target variables. However, `loo_val_targets` "
                "has second dimension greater than 1."
            )
        if loo_test_outputs.ndim < 2:
            raise ValueError("`loo_test_outputs` must have at least two dimensions.")
        elif loo_test_outputs.ndim == 2:
            loo_test_outputs = loo_test_outputs[:, :, None]
        elif loo_test_outputs.shape[2] != 1:
            raise ValueError(
                "This method is supported only for scalar model outputs only. However, `loo_test_outputs` "
                "has last dimension greater than 1."
            )

        r = jnp.abs(loo_val_targets - loo_val_outputs)
        q = jnp.quantile(r, q=1 - error, axis=0)
        qleft = jnp.min(loo_test_outputs, axis=0) - q
        qright = jnp.max(loo_test_outputs, axis=0) + q
        return jnp.array(list(zip(qleft, qright))).squeeze(2)
