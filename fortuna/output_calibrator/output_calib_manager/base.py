from typing import (
    Optional,
    Tuple,
    Union,
)

from flax.core import FrozenDict
import flax.linen as nn
from flax.training.checkpoints import PyTree
from jax import random
from jax._src.prng import PRNGKeyArray
import jax.numpy as jnp

from fortuna.typing import (
    Array,
    CalibMutable,
    CalibParams,
)
from fortuna.utils.random import WithRNG


class OutputCalibManager(WithRNG):
    def __init__(self, output_calibrator: Optional[nn.Module] = None):
        self.output_calibrator = output_calibrator

    def apply(
        self,
        params: CalibParams,
        outputs: Array,
        mutable: Optional[CalibMutable] = None,
        calib: bool = False,
        rng: Optional[PRNGKeyArray] = None,
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, PyTree]]:
        """
        Apply the output calibrator forward pass.

        Parameters
        ----------
        params : Params
            The random parameters of the probabilistic model.
        outputs : Array
            Outputs for each data point.
        mutable : Optional[Mutable]
            The mutable objects used to evaluate the models.
        calib : bool
            Whether the method is called during calibration.
        rng: Optional[PRNGKeyArray]
            A random number generator.
            If not passed,
            this will be taken from the attributes of this class.

        Returns
        -------
        Union[jnp.ndarray, Tuple[jnp.ndarray, PyTree]]
            The output of the model manager for each input. Mutable objects may also be returned.
        """
        if self.output_calibrator is None:
            return outputs
        if mutable is None:
            mutable = False
        variables = params.unfreeze()

        # setup dropout key
        if rng is not None:
            rng, dropout_rng = random.split(rng, 2)
            rngs = {"dropout": dropout_rng}
        else:
            rngs = None

        if mutable:
            mutable_variables = mutable.unfreeze()
            variables.update(mutable_variables)
            mutable = list(mutable_variables.keys())
        if calib and mutable:
            outputs, mutable = self.output_calibrator.apply(
                variables, outputs, mutable=mutable, train=calib, rngs=rngs
            )
            return outputs, {"mutable": mutable}
        else:
            return self.output_calibrator.apply(
                variables, outputs, train=calib, mutable=False, rngs=rngs
            )

    def init(
        self, output_dim: int, rng: Optional[PRNGKeyArray] = None, **kwargs
    ) -> Optional[FrozenDict]:
        """
        Initialize random parameters and mutable objects.

        Parameters
        ----------
        output_dim: int
            The output dimension.
        rng: Optional[PRNGKeyArray]
            A random number generator.
            If not passed,
            this will be taken from the attributes of this class.

        Returns
        -------
        Optional[FrozenDict]
            Initialized random parameters and mutable objects.
        """
        if rng is None:
            rng = self.rng.get()
        rng, params_key, dropout_key = random.split(rng, 3)
        rngs = {"params": params_key, "dropout": dropout_key}
        return (
            self.output_calibrator.init(rngs, jnp.zeros((1, output_dim)), **kwargs)
            if self.output_calibrator is not None
            else None
        )
