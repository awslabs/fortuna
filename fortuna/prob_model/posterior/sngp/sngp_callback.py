import jax.numpy as jnp
from flax.core import FrozenDict

from fortuna.training.callback import Callback
from fortuna.training.train_state import TrainState
from fortuna.utils.nested_dicts import find_one_path_to_key, nested_get, nested_update


class ResetCovarianceCallback(Callback):
    """
    Reset, at the beginning of each epoch, the covariance matrix estimated while training an SNGP model.
    """

    def __init__(self, precision_matrix_key_name: str, ridge_penalty: float):
        self.precision_matrix_key_name = precision_matrix_key_name
        self.ridge_penalty = ridge_penalty

    def training_epoch_start(self, state: TrainState) -> TrainState:
        key_paths = find_one_path_to_key(state.mutable, self.precision_matrix_key_name)
        precision_matrix = nested_get(state.mutable, key_paths)
        if precision_matrix.ndim == 2:
            n, _ = precision_matrix.shape  # rows, cols
            init_precision_matrix = (
                jnp.eye(n, dtype=precision_matrix.dtype) * self.ridge_penalty
            )
        elif precision_matrix.ndim == 3:
            d, n, _ = precision_matrix.shape  # num_devices, rows, cols
            init_precision_matrix = (
                jnp.eye(n, dtype=precision_matrix.dtype) * self.ridge_penalty
            )
            init_precision_matrix = jnp.broadcast_to(init_precision_matrix, (d, n, n))

        partially_updated_mutables = init_precision_matrix
        for key in reversed(key_paths):
            partially_updated_mutables = {key: partially_updated_mutables}
        mutables = nested_update(state.mutable.unfreeze(), partially_updated_mutables)
        mutables = FrozenDict(mutables)
        return state.replace(mutable=mutables)
