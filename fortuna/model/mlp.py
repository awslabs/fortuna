from typing import Callable, Optional, Tuple

import flax.linen as nn
import jax.numpy as jnp

from fortuna.typing import Array


class MLP(nn.Module):
    """
    A multi-layer perceptron (MLP).

    Parameters
    ----------
    output_dim: int
        The output model dimension.
    widths: Tuple[int]
        The number of units of each hidden layer. Default: (30, 30).
    activations: Tuple[Callable[[Array], Array]]
        The activation functions after each hidden layer. Default: (flax.linen.relu, flax.linen.relu).
    """

    output_dim: int
    widths: Optional[Tuple[int]] = (30, 30)
    activations: Optional[Tuple[Callable[[Array], Array]]] = (nn.relu, nn.relu)

    def setup(self):
        if len(self.widths) != len(self.activations):
            raise Exception(
                "`widths` and `activations` must have the same number of elements."
            )
        self.dfe_subnet = MLPDeepFeatureExtractorSubNet(
            widths=self.widths,
            activations=self.activations[:-1],
        )
        self.output_subnet = MLPOutputSubNet(
            activation=self.activations[-1], output_dim=self.output_dim
        )

    def __call__(self, x: Array, **kwargs) -> jnp.ndarray:
        x = self.dfe_subnet(x)
        x = self.output_subnet(x)
        return x


class MLPDeepFeatureExtractorSubNet(nn.Module):
    widths: Tuple[int]
    activations: Tuple[Callable[[Array], Array]]
    """
    MLP Deep feature extractor sub-network.

    Attributes
    ----------
    widths: Tuple[int]
        The number of units of each hidden layer.
    activations: Tuple[Callable[[Array], Array]]
        The activation functions after each hidden layer. 
    """

    @nn.compact
    def __call__(self, x: Array, **kwargs) -> jnp.ndarray:
        """
        Forward pass.

        Parameters
        ----------
        x: Array
            Inputs.

        Returns
        -------
        jnp.ndarray
            Output of the hidden layers.
        """
        n_activations = len(self.activations)

        def update(i: int, x):
            x = nn.Dense(self.widths[i], name="hidden" + str(i + 1))(x)
            if i < n_activations:
                x = self.activations[i](x)
            return x

        x = x.reshape(x.shape[0], -1)
        for i in range(0, len(self.widths)):
            x = update(i, x)
        return x


class MLPOutputSubNet(nn.Module):
    output_dim: int
    activation: Optional[Callable[[Array], Array]] = None
    """
    MLP output sub-network.

    Attributes
    ----------
    output_dim: int
        The output model dimension.
    activations: Tuple[Callable[[Array], Array]]
        The activation functions after each hidden layer. 
    """

    @nn.compact
    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """
        Forward pass.

        Parameters
        ----------
        x: jnp.ndarray
            Outputs of the hidden layers.

        Returns
        -------
        jnp.ndarray
            Output of the last layer.
        """
        if self.activation is not None:
            x = self.activation(x)
        x = nn.Dense(self.output_dim, name="last")(x)
        return x
