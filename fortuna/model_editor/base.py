from typing import (
    Any,
    Callable,
    Dict,
    Tuple,
    Union,
)

import flax.linen as nn
import jax.numpy as jnp

from fortuna.typing import (
    InputData,
    Params,
)


class ModelEditor(nn.Module):
    @nn.compact
    def __call__(
        self,
        apply_fn: Callable[
            [Params, InputData], Union[jnp.ndarray, Tuple[jnp.ndarray, Dict]]
        ],
        model_params: Params,
        x: Any,
        has_aux: bool,
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Dict]]:
        """
        Apply a transformation to the forward pass.

        Parameters
        ----------
        apply_fn: Callable[[Params, InputData], Union[jnp.ndarray, Tuple[jnp.ndarray, Dict]]]
            The model forward pass.
        model_params: Params
            The model parameters.
        x: Array
            Batch of inputs.
        has_aux: bool
            Whether the forward pass returns auxiliary objects.

        Returns
        -------
        Union[jnp.ndarray, Tuple[jnp.ndarray, Dict]]
            Return the transformed outputs, and auxiliary objects if available.
        """
        pass
