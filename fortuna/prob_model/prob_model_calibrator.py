from typing import Any, Callable, Dict, Tuple, Union

import jax
import jax.numpy as jnp
from flax import jax_utils
from jax._src.prng import PRNGKeyArray
from jax.tree_util import tree_map

from fortuna.training.output_calibrator import (OutputCalibratorABC, JittedMixin,
                                                MultiDeviceMixin)
from fortuna.output_calib_model.state import OutputCalibState
from fortuna.data import TargetsLoader
from fortuna.typing import Array, Batch, CalibMutable, CalibParams


class ProbModelOutputCalibrator(OutputCalibratorABC):
    def training_loss_step(
        self,
        loss_fun: Callable[[Any], Union[float, Tuple[float, dict]]],
        params: CalibParams,
        batch: Batch,
        outputs: Array,
        mutable: CalibMutable,
        rng: PRNGKeyArray,
        n_data: int,
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        return_aux = ["outputs"]
        if mutable is not None:
            return_aux += ["mutable"]
        loss, aux = loss_fun(
            batch,
            n_data=n_data,
            return_aux=["outputs", "calib_mutable"],
            ensemble_outputs=outputs,
            calib_params=params,
            calib_mutable=mutable,
            rng=rng,
        )
        logging_kwargs = None
        return (
            loss,
            {
                "outputs": aux.get("outputs"),
                "mutable": aux.get("calib_mutable"),
                "logging_kwargs": logging_kwargs,
            },
        )

    def val_loss_step(
        self,
        state: OutputCalibState,
        batch: Batch,
        outputs: Array,
        loss_fun: Callable,
        rng: PRNGKeyArray,
        n_data: int,
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        loss, aux = loss_fun(
            batch,
            n_data=n_data,
            return_aux=["outputs"],
            ensemble_outputs=outputs,
            calib_params=state.params,
            calib_mutable=state.mutable,
            rng=rng,
        )
        return loss, aux

    def __str__(self):
        return "calibration"


class ProbModelMultiDeviceMixin(MultiDeviceMixin):
    @staticmethod
    def _add_device_dim_to_outputs_loader(
        outputs_loader: TargetsLoader,
    ) -> TargetsLoader:
        def _reshape_batch(batch):
            n_devices = jax.local_device_count()
            if batch.shape[1] % n_devices != 0:
                raise ValueError(
                    f"The size of all output batches must be a multiple of {n_devices}, that is the number of "
                    f"available devices. However, a batch of outputs with shape {batch.shape[1]} was found. "
                    f"Please set an appropriate batch size."
                )
            shape = batch.shape
            return (
                batch.swapaxes(0, 1)
                .reshape(n_devices, shape[1] // n_devices, shape[0], shape[2])
                .swapaxes(1, 2)
            )

        class TargetsLoaderWrapper:
            def __init__(self, outputs_loader: TargetsLoader):
                self._outputs_loader = outputs_loader

            def __iter__(self):
                outputs_loader = map(
                    lambda batch: tree_map(_reshape_batch, batch), self._outputs_loader
                )
                outputs_loader = jax_utils.prefetch_to_device(outputs_loader, 2)
                yield from outputs_loader

        return (
            TargetsLoaderWrapper(outputs_loader)
            if outputs_loader is not None
            else outputs_loader
        )


class JittedProbModelOutputCalibrator(JittedMixin, ProbModelOutputCalibrator):
    pass


class MultiDeviceProbModelOutputCalibrator(ProbModelMultiDeviceMixin, ProbModelOutputCalibrator):
    pass
