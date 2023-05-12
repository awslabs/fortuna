import pathlib
from typing import (
    Dict,
    Iterable,
    Tuple,
    Union,
)

from flax.core import FrozenDict
import jax.numpy as jnp
import numpy as np
from optax._src.base import (
    GradientTransformation,
    PyTree,
)

Params = FrozenDict[str, FrozenDict[str, PyTree]]
Mutable = FrozenDict[str, FrozenDict[str, PyTree]]
CalibParams = FrozenDict[str, PyTree]
CalibMutable = FrozenDict[str, PyTree]
OptaxOptimizer = GradientTransformation
Array = Union[jnp.ndarray, np.ndarray]
Status = Dict[str, Array]
Path = Union[str, pathlib.Path]
InputData = Array
Targets = Array
Batch = Tuple[InputData, Targets]
Outputs = jnp.ndarray
Uncertainties = jnp.ndarray
Predictions = jnp.ndarray
AnyKey = Union[str, int]
Shape = Iterable[int]
