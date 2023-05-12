from typing import (
    Any,
    Callable,
    Optional,
    Tuple,
    Union,
)

import flax.linen as nn
import jax.numpy as jnp

from fortuna.typing import Array

ModuleDef = Any


class MLP(nn.Module):
    """
    A multi-layer perceptron (MLP).

    Attributes
    ----------
    output_dim: int
        The output model dimension.
    widths: Tuple[int]
        The number of units of each hidden layer. Default: (30, 30).
    activations: Tuple[Callable[[Array], Array]]
        The activation functions after each hidden layer. Default: (flax.linen.relu, flax.linen.relu).
    dropout: ModuleDef
        Dropout module.
    dropout_rate: float
        Dropout rate.
    dense: ModuleDef
        Dense module.
    """

    output_dim: int
    widths: Optional[Tuple[int]] = (30, 30)
    activations: Optional[Tuple[Callable[[Array], Array]]] = (nn.relu, nn.relu)
    dropout: ModuleDef = nn.Dropout
    dropout_rate: float = 0.0
    dense: ModuleDef = nn.Dense

    def setup(self):
        if len(self.widths) != len(self.activations):
            raise Exception(
                "`widths` and `activations` must have the same number of elements."
            )
        self.dfe_subnet = MLPDeepFeatureExtractorSubNet(
            dense=self.dense,
            widths=self.widths,
            activations=self.activations[:-1],
            dropout=self.dropout,
            dropout_rate=self.dropout_rate,
        )
        self.output_subnet = MLPOutputSubNet(
            dense=self.dense,
            activation=self.activations[-1],
            output_dim=self.output_dim,
        )

    def __call__(self, x: Array, train: bool = False, **kwargs) -> jnp.ndarray:
        x = self.dfe_subnet(x, train)
        x = self.output_subnet(x)
        return x


class DeepResidualNet(MLP):
    """
    A multi-layer perceptron with residual connections
    """

    def setup(self):
        if len(self.widths) != len(self.activations):
            raise Exception(
                "`widths` and `activations` must have the same number of elements."
            )
        self.dfe_subnet = DeepResidualFeatureExtractorSubNet(
            dense=self.dense,
            widths=self.widths,
            activations=self.activations[:-1],
            dropout=self.dropout,
            dropout_rate=self.dropout_rate,
        )
        self.output_subnet = MLPOutputSubNet(
            dense=self.dense,
            activation=self.activations[-1],
            output_dim=self.output_dim,
        )


class MLPDeepFeatureExtractorSubNet(nn.Module):
    widths: Tuple[int]
    activations: Tuple[Callable[[Array], Array]]
    dense: ModuleDef = nn.Dense
    dropout: ModuleDef = nn.Dropout
    dropout_rate: float = 0.0

    """
    MLP Deep feature extractor sub-network.

    Attributes
    ----------
    widths: Tuple[int]
        The number of units of each hidden layer.
    activations: Tuple[Callable[[Array], Array]]
        The activation functions after each hidden layer.
    dense: ModuleDef
        Dense module.
    dropout: ModuleDef
        Dropout module.
    dropout_rate: float
        Dropout rate.
    """

    @nn.compact
    def __call__(self, x: Array, train: bool = False, **kwargs) -> jnp.ndarray:
        """
        Forward pass.

        Parameters
        ----------
        x: Array
            Inputs.
        train: bool
            Whether it is training or inference.

        Returns
        -------
        jnp.ndarray
            Output of the hidden layers.
        """
        if hasattr(self, "spectral_norm"):
            dense = self.spectral_norm(self.dense, train=train)
        else:
            dense = self.dense
        dropout = self.dropout(self.dropout_rate)
        n_activations = len(self.activations)

        def update(i: int, x):
            x = dense(self.widths[i], name="hidden" + str(i + 1))(x)
            if i < n_activations:
                x = self.activations[i](x)
            x = dropout(x, deterministic=not train)
            return x

        x = x.reshape(x.shape[0], -1)
        for i in range(0, len(self.widths)):
            x = update(i, x)
        return x


class DeepResidualFeatureExtractorSubNet(MLPDeepFeatureExtractorSubNet):
    @nn.compact
    def __call__(self, x: Array, train: bool = False, **kwargs) -> jnp.ndarray:
        """
        Forward pass.

        Parameters
        ----------
        x: Array
            Inputs.
        train: bool
            Whether it is training or prediction.

        Returns
        -------
        jnp.ndarray
            Output of the hidden layers.
        """
        if hasattr(self, "spectral_norm"):
            dense = self.spectral_norm(self.dense, train=train)
        else:
            dense = self.dense
        dropout = self.dropout(self.dropout_rate)
        n_activations = len(self.activations)

        def update(i: int, x):
            x = dense(self.widths[i], name="hidden" + str(i + 1))(x)
            if i < n_activations:
                x = self.activations[i](x)
            x = dropout(x, deterministic=not train)
            return x

        x = x.reshape(x.shape[0], -1)
        x = dense(self.widths[0], name="hidden" + str(0 + 1))(x)
        for i in range(1, len(self.widths)):
            h = jnp.copy(x)
            x = update(i, x)
            x = h + x
        return x


class MLPOutputSubNet(nn.Module):
    output_dim: int
    activation: Optional[Callable[[Array], Array]] = None
    dense: ModuleDef = nn.Dense
    """
    MLP output sub-network.

    Attributes
    ----------
    output_dim: int
        The output model dimension.
    activations: Tuple[Callable[[Array], Array]]
        The activation functions after each hidden layer.
    dense: ModuleDef
        Dense module.
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
        x = self.dense(self.output_dim, name="last")(x)
        return x
