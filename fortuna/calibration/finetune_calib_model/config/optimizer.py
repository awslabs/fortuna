from typing import Optional

import optax

from fortuna.typing import OptaxOptimizer


class Optimizer:
    def __init__(
        self,
        method: Optional[OptaxOptimizer] = optax.adam(1e-2),
        n_epochs: int = 100,
    ):
        """
        An object to configure the optimization in the calibration process.

        Parameters
        ----------
        method: OptaxOptimizer
            An Optax optimizer.
        n_epochs: int
            Maximum number of epochs to run the calibration for.
        """
        self.method = method
        self.n_epochs = n_epochs
