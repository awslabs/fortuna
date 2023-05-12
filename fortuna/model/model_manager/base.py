import abc
from typing import Dict, Optional, Tuple, Union, Mapping

import jax.numpy as jnp
from flax import linen as nn
from flax.core import FrozenDict
from flax.training.checkpoints import PyTree
from jax._src.prng import PRNGKeyArray

from fortuna.typing import Mutable, Params, InputData
from fortuna.utils.random import WithRNG


class ModelManager(WithRNG, abc.ABC):
    """
    Abstract model manager class.
    It orchestrates the forward pass of the models in the probabilistic model.
    """

    def __init__(self, model: nn.Module):
        self.model = model

    @abc.abstractmethod
    def apply(
        self,
        params: Params,
        inputs: InputData,
        mutable: Optional[Mutable] = None,
        train: bool = False,
        rng: Optional[PRNGKeyArray] = None,
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, PyTree]]:
        """
        Apply the models' forward pass.

        Parameters
        ----------
        params : Params
            The random parameters of the probabilistic model.
        inputs : InputData
            Input data points.
        mutable : Optional[Mutable]
            The mutable objects used to evaluate the models.
        train : bool
            Whether the method is called during training.
        rng: Optional[PRNGKeyArray]
            A random number generator.
            If not passed,
            this will be taken from the attributes of this class.

        Returns
        -------
        Union[jnp.ndarray, Tuple[jnp.ndarray, PyTree]]
            The output of the model manager for each input. Mutable objects may also be returned.
        """
        pass

    @abc.abstractmethod
    def init(
        self, input_shape: Tuple[int, ...], rng: Optional[PRNGKeyArray] = None, **kwargs
    ) -> Dict[str, Mapping]:
        """
        Initialize random parameters and mutable objects.

        Parameters
        ----------
        input_shape : Tuple
            The shape of the input variable.
        rng: Optional[PRNGKeyArray]
            A random number generator.
            If not passed,
            this will be taken from the attributes of this class.

        Returns
        -------
        Dict[str, FrozenDict]
            Initialized random parameters and mutable objects.
        """
        pass
