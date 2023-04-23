import pathlib
from typing import Dict, Tuple, Union

import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict
from optax._src.base import GradientTransformation, PyTree

Params = FrozenDict[str, FrozenDict[str, PyTree]]
Mutable = FrozenDict[str, FrozenDict[str, PyTree]]
CalibParams = FrozenDict[str, PyTree]
CalibMutable = FrozenDict[str, PyTree]
OptaxOptimizer = GradientTransformation
Array = Union[jnp.ndarray, np.ndarray]
Status = Dict[str, Array]
Path = Union[str, pathlib.Path]
Batch = Tuple[Array, Array]
Outputs = jnp.ndarray
Targets = Array
Uncertainties = jnp.ndarray
Predictions = jnp.ndarray
