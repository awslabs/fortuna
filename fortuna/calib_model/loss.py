from typing import (
    Any,
    Callable,
    List,
    Optional,
    Tuple,
)

from jax._src.prng import PRNGKeyArray
import jax.numpy as jnp

from fortuna.likelihood.base import Likelihood
from fortuna.typing import (
    Batch,
    CalibMutable,
    CalibParams,
    Mutable,
    Params,
    Targets,
)
from fortuna.utils.random import WithRNG


class Loss(WithRNG):
    def __init__(
        self,
        likelihood: Likelihood,
        loss_fn: Callable[[Callable, Targets], jnp.ndarray],
    ):
        self.likelihood = likelihood
        self.loss_fn = loss_fn

    def __call__(
        self,
        params: Params,
        batch: Batch,
        mutable: Optional[Mutable] = None,
        calib_params: Optional[CalibParams] = None,
        calib_mutable: Optional[CalibMutable] = None,
        return_aux: Optional[List[str]] = None,
        train: bool = False,
        outputs: Optional[jnp.ndarray] = None,
        rng: Optional[PRNGKeyArray] = None,
        **kwargs,
    ) -> Tuple[jnp.ndarray, Any]:
        if return_aux is None:
            return_aux = []
        supported_aux = ["outputs", "mutable", "calib_mutable"]
        unsupported_aux = [s for s in return_aux if s not in supported_aux]
        if sum(unsupported_aux) > 0:
            raise AttributeError(
                """The auxiliary objects {} is unknown. Please make sure that all elements of `return_aux`
                            belong to the following list: {}""".format(
                    unsupported_aux, supported_aux
                )
            )
        if train and outputs is not None:
            raise ValueError(
                """When `outputs` is available, `train` must be set to `False`."""
            )
        if "mutable" in return_aux and outputs is not None:
            raise ValueError(
                """When `outputs` is available, `return_aux` cannot contain 'mutable'`."""
            )
        if not train and "mutable" in return_aux:
            raise ValueError(
                "Returning an auxiliary mutable is supported only during training. Please either set `train` to "
                "`True`, or remove 'mutable' from `return_aux`."
            )
        if "mutable" in return_aux and mutable is None:
            raise ValueError(
                "In order to be able to return an auxiliary mutable, an initial mutable must be passed as `mutable`. "
                "Please either remove 'mutable' from `return_aux`, or pass an initial mutable as `mutable`."
            )
        if "mutable" not in return_aux and mutable is not None and train is True:
            raise ValueError(
                """You need to add `mutable` to `return_aux`. When you provide a (not null) `mutable` variable during
                training, that variable will be updated during the forward pass."""
            )

        if outputs is None:
            outs = self.likelihood.model_manager.apply(
                params,
                batch[0],
                train=train,
                mutable=mutable,
                rng=rng,
            )
            if train and "mutable" is not None:
                outputs, aux = outs
                mutable = aux["mutable"]
            else:
                outputs = outs

        aux = dict()
        if self.likelihood.output_calib_manager is not None:
            outs = self.likelihood.output_calib_manager.apply(
                params=calib_params["output_calibrator"]
                if calib_params is not None
                else None,
                mutable=calib_mutable["output_calibrator"]
                if calib_mutable is not None
                else None,
                outputs=outputs,
                calib="calib_mutable" in return_aux,
            )
            if (
                calib_mutable is not None
                and calib_mutable["output_calibrator"] is not None
                and "calib_mutable" in return_aux
            ):
                outputs, aux["calib_mutable"] = outs
                aux["calib_mutable"] = dict(output_calibrator=aux["calib_mutable"])
            else:
                outputs = outs
                if "calib_mutable" in return_aux:
                    aux["calib_mutable"] = dict(output_calibrator=None)

        if "outputs" in return_aux:
            aux["outputs"] = outputs
        if "mutable" in return_aux:
            aux["mutable"] = mutable

        return self.loss_fn(outputs, batch[1]), aux
