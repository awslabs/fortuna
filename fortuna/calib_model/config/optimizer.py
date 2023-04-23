from typing import Optional, Callable, Tuple

import optax

from fortuna.typing import OptaxOptimizer
import jax.numpy as jnp


class Optimizer:
    def __init__(
        self,
        method: Optional[OptaxOptimizer] = optax.adam(1e-2),
        n_epochs: int = 100,
        freeze_map: Optional[Callable[[Tuple[str], jnp.ndarray], str]] = None
    ):
        """
        An object to configure the optimization in the calibration process.

        Parameters
        ----------
        method: OptaxOptimizer
            An Optax optimizer.
        n_epochs: int
            Maximum number of epochs to run the calibration for.
        freeze_map: Optional[Callable[[Tuple[str], jnp.ndarray], str]]
            A callable taking in input a path in the nested dictionary of parameters, as well as the corresponding
            array of parameters, and returns "trainable" or "freeze", according to whether the corresponding parameter
            should be optimized or not.

            Examples
            --------
            .. code-block:: python
                def freeze_map(path: Tuple[str], v: Array) -> str:
                    path = [p[:6] for p in path]  # take only the first 6 characters of each key"
                    return 'trainable' if "Dense" in path else 'frozen'`
        """
        self.method = method
        self.n_epochs = n_epochs
        self.freeze_map = freeze_map
