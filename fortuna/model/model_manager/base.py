import abc
from typing import (
    Dict,
    Mapping,
    Optional,
    Tuple,
    Union,
)

from flax import linen as nn
from flax.core import FrozenDict
from jax._src.prng import PRNGKeyArray
import jax.numpy as jnp
from optax._src.base import PyTree

from fortuna.typing import (
    InputData,
    Mutable,
    Params,
    Shape,
)
from fortuna.utils.random import WithRNG


class ModelManager(WithRNG, abc.ABC):
    """
    Abstract model manager class.
    It orchestrates the forward pass of the models in the probabilistic model.
    """

    def __init__(self, model: nn.Module, model_editor: Optional[nn.Module] = None):
        self.model = model
        self.model_editor = model_editor

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
        self, input_shape: Shape, rng: Optional[PRNGKeyArray] = None, **kwargs
    ) -> Dict[str, Mapping]:
        """
        Initialize random parameters and mutable objects.

        Parameters
        ----------
        input_shape : Shape
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
