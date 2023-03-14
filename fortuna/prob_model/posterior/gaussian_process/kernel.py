import jax.numpy as jnp
import jax.random as jr
import jax
import flax
from fortuna.typing import Params
from fortuna.training.train_state import TrainState
import flax.linen as nn
import typing as tp
import matplotlib.pyplot as plt

key = jr.PRNGKey(123)


class AbstractKernel:
    pass


class RBF(nn.Module):
    r"""The radial basis function (RBF) kernel. Also known as the squared exponential.

    The kernel is parameterised by a lengthscale :math:`\ell\in\mathbb{R}_{>0}` and variance :math:`\sigma^2\in\mathbb{R}_{>0}`.
    It is defined as:
    .. math::
       :name: rbf_kernel

        k(x, y) = \sigma^2 \exp\left(-\frac{1}{2} \sum_{i=1}^D \frac{(x_i - y_i)^2}{\ell_i^2}\right)
    """

    lengthscale: jax.Array = jnp.array([1.0])
    variance: jax.Array = jnp.array([1.0])

    def kernel_fn(self, xi, yj):
        return jnp.exp(-0.5 * jnp.sum((xi - yj) ** 2))

    @nn.compact
    def __call__(self, x, y=None):
        lengthscale = self.param(
            "ell", nn.initializers.ones_init(), (x.shape[1],)
        )
        variance = self.param("sigma2", nn.initializers.ones_init(), (1,))
        if not y:
            y = x

        # Scale the inputs
        x /= lengthscale
        y /= lengthscale

        # Evaluate the kernel
        Kxy = jax.vmap(lambda x: jax.vmap(lambda y: self.kernel_fn(x, y))(y))(x)

        # Scale the kernel
        Kxy *= variance

        return Kxy
