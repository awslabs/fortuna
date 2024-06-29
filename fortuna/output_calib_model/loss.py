from typing import (
    Any,
    Callable,
    List,
    Optional,
    Tuple,
    Union,
)

import jax
import jax.numpy as jnp

from fortuna.output_calib_model.predictive.base import Predictive
from fortuna.typing import (
    Array,
    CalibMutable,
    CalibParams,
    Outputs,
    Targets,
)
from fortuna.utils.random import WithRNG


class Loss(WithRNG):
    def __init__(
        self, predictive: Predictive, loss_fn: Callable[[Outputs, Targets], jnp.ndarray]
    ):
        self.predictive = predictive
        self.loss_fn = loss_fn

    def __call__(
        self,
        params: CalibParams,
        targets: Array,
        outputs: Array,
        mutable: Optional[CalibMutable] = None,
        rng: Optional[jax.Array] = None,
        return_aux: Optional[List[str]] = None,
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Any]]:
        if return_aux is None:
            return_aux = []
        supported_aux = ["outputs", "mutable"]
        unsupported_aux = [s for s in return_aux if s not in supported_aux]
        if sum(unsupported_aux) > 0:
            raise AttributeError(
                """The auxiliary objects {} is unknown. Please make sure that all elements of `return_aux`
                            belong to the following list: {}""".format(
                    unsupported_aux, supported_aux
                )
            )
        aux = dict()
        outs = self.predictive.output_calib_manager.apply(
            params=params["output_calibrator"],
            outputs=outputs,
            mutable=mutable["output_calibrator"],
            rng=rng,
            calib="mutable" in return_aux,
        )
        if (
            mutable is not None
            and mutable["output_calibrator"] is not None
            and "mutable" in return_aux
        ):
            outputs, aux["mutable"] = outs
            aux["mutable"] = dict(output_calibrator=aux["mutable"])
        else:
            outputs = outs
            if "mutable" in return_aux:
                aux["mutable"] = dict(output_calibrator=None)

        if "outputs" in return_aux:
            aux["outputs"] = outputs

        loss = self.loss_fn(outputs, targets)

        if len(return_aux) == 0:
            return loss
        else:
            return loss, aux
