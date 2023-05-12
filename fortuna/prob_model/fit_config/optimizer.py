from typing import (
    Callable,
    Optional,
    Tuple,
)

import optax

from fortuna.typing import (
    AnyKey,
    Array,
    OptaxOptimizer,
)


class FitOptimizer:
    def __init__(
        self,
        method: Optional[OptaxOptimizer] = optax.adam(1e-3),
        n_epochs: int = 100,
        freeze_fun: Optional[Callable[[Tuple[AnyKey, ...], Array], str]] = None,
    ):
        """
        An object to configure the optimization in the posterior fitting.

        Parameters
        ----------
        method: OptaxOptimizer
            An Optax optimizer.
        n_epochs: int
            Maximum number of epochs to run the training for.
        freeze_fun: Optional[Callable[[Tuple[AnyKey, ...], Array], str]]
            A callable taking in input a path in the nested dictionary of parameters, as well as the corresponding
            array of parameters, and returns "trainable" or "freeze", according to whether the corresponding parameter
            should be optimized or not.
        """
        self.method = method
        self.n_epochs = n_epochs
        self.freeze_fun = freeze_fun
