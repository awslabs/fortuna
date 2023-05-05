from jax.tree_util import tree_map
import numpy as np
import jax.numpy as jnp

from fortuna.data.loader import DataLoader
from fortuna.typing import InputData, Shape


def check_data_loader_is_not_random(data_loader: DataLoader) -> None:
    flag = False
    for (x1, y1), (x2, y2) in zip(data_loader, data_loader):
        if not np.alltrue(x1 == x2) or not np.alltrue(y1 == y2):
            flag = True
            break
    if flag:
        raise ValueError(
            """The data loader randomizes at every iteration. To perform this method, please provide a data loader that 
            generates the same sequence of data when called multiple times."""
        )


def get_input_shape(inputs: InputData) -> Shape:
    return tree_map(lambda x: x.shape[1:], inputs)


def get_inputs_from_shape(input_shape: Shape) -> InputData:
    if isinstance(input_shape, tuple):
        inputs = jnp.zeros((1,) + input_shape)
    elif isinstance(input_shape, dict):
        inputs = {k: jnp.zeros((1,) + v) for k, v in input_shape.items()}
    else:
        raise ValueError("Data batches shape have to be of type fortuna.typing.Shape")
    return inputs
