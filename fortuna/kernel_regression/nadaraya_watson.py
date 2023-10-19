from typing import Callable

from jax import vmap
import jax.numpy as jnp

from fortuna.kernel_regression.kernels.gaussian import gaussian_kernel
from fortuna.typing import Array


class NadarayaWatsonKernelRegressor:
    def __init__(
        self,
        train_inputs: Array,
        train_targets: Array,
        kernel: Callable[[Array], Array] = gaussian_kernel,
    ):
        """
        A Nadaraya-Watson kernel regressor.

        Parameters
        ----------
        kernel: Callable[[Array], Array]
            A kernel.
        train_inputs: Array
            Training inputs.
        train_targets: Array
            Training targets.
        """
        self.kernel = kernel

        if (train_inputs.ndim > 1) or (train_targets.ndim > 1):
            raise ValueError(
                "Both `train_inputs` and `train_targets` must be one-dimensional arrays."
            )
        if len(train_inputs) != len(train_targets):
            raise ValueError(
                "`train_inputs` and `train_targets` must have the same length."
            )
        self.train_inputs = jnp.copy(train_inputs)
        self._mean_train_targets = jnp.mean(train_targets)
        self._std_train_targets = jnp.std(train_targets)
        self.standardized_train_targets = (
            jnp.copy(train_targets) - self._mean_train_targets
        ) / self._std_train_targets

    def predict(self, inputs: Array, bandwidth: float = 0.1):
        """
        Predict the target

        Parameters
        ----------
        inputs: Array
            Inputs.
        bandwidth: float
            Kernel bandwidth.

        Returns
        -------
        Predictions for the given inputs.
        """
        """"""
        if inputs.ndim > 1:
            raise ValueError("`inputs` must be a one-dimensional array.")
        inputs = jnp.copy(inputs)
        kernels = vmap(
            lambda x: self.evaluate_scaled_kernel(
                x - self.train_inputs, bandwidth=bandwidth
            )
        )(inputs)
        m = jnp.sum(kernels * self.standardized_train_targets[None], axis=1) / jnp.sum(
            kernels, axis=1
        )
        m *= self._std_train_targets
        m += self._mean_train_targets
        return m

    def evaluate_scaled_kernel(self, inputs: Array, bandwidth: float) -> Array:
        """
        Given a kernel :math:`K(x)`, Evaluate the scaled kernel :math:`K_h(x) := \frac{1}{h}K\left(\frac{x}{h}\right)`.

        Parameters
        ----------
        inputs: Array
            Inputs.
        bandwidth: Array
            Bandwidth.

        Returns
        -------
        Evaluation of the scaled kernel.
        """
        return self.kernel(inputs / bandwidth) / bandwidth
