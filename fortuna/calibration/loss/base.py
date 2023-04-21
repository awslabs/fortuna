import abc
from fortuna.typing import Array, Params, CalibParams, Mutable, CalibMutable
from typing import List, Optional, Tuple, Union, Any
from jax._src.prng import PRNGKeyArray
import jax.numpy as jnp


class Loss:
    @abc.abstractmethod
    def __call__(
        self,
        params: Union[Params, CalibParams],
        targets: Array,
        outputs: Array,
        mutable: Optional[Union[Mutable, CalibMutable]] = None,
        rng: Optional[PRNGKeyArray] = None,
        return_aux: Optional[List[str]] = None,
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Any]]:
        pass
