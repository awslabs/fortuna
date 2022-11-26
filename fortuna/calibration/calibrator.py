import abc
import collections
import logging
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from flax import jax_utils
from flax.training.common_utils import stack_forest
from jax import lax, random, value_and_grad
from jax._src.prng import PRNGKeyArray
from jax.tree_util import tree_map
from tqdm import trange
from tqdm.std import tqdm as TqdmDecorator

from fortuna.calibration.state import CalibState
from fortuna.data.loader import DataLoader, TargetsLoader
from fortuna.training.mixin import (InputValidatorMixin,
                                    WithCheckpointingMixin,
                                    WithEarlyStoppingMixin)
from fortuna.typing import (Array, Batch, CalibMutable, CalibParams, Path,
                            Status)
from fortuna.utils.builtins import HashableMixin


class CalibratorABC(
    HashableMixin,
    WithCheckpointingMixin,
    WithEarlyStoppingMixin,
    InputValidatorMixin,
    metaclass=abc.ABCMeta,
):
    def __init__(
        self,
        *args,
        calib_outputs_loader: TargetsLoader,
        predict_fn: Callable[[jnp.ndarray], jnp.ndarray],
        uncertainty_fn: Callable[[jnp.ndarray], jnp.ndarray],
        val_outputs_loader: Optional[TargetsLoader] = None,
        save_checkpoint_dir: Optional[Path] = None,
        save_every_n_steps: Optional[int] = None,
        keep_top_n_checkpoints: int = 2,
        disable_training_metrics_computation: bool = False,
        eval_every_n_epochs: int = 1,
        **kwargs,
    ):
        super(CalibratorABC, self).__init__(*args, **kwargs)
        self._calib_outputs_loader = calib_outputs_loader
        self._val_outputs_loader = val_outputs_loader
        self.predict_fn = predict_fn
        self.uncertainty_fn = uncertainty_fn
        self.save_checkpoint_dir = save_checkpoint_dir
        self.save_every_n_steps = save_every_n_steps
        self.keep_top_n_checkpoints = keep_top_n_checkpoints
        self.disable_training_metrics_computation = disable_training_metrics_computation
        self.eval_every_n_epochs = eval_every_n_epochs
        self.multi_gpu = False

    def train(
        self,
        rng: PRNGKeyArray,
        state: CalibState,
        fun: Callable,
        training_data_loader: DataLoader,
        training_dataset_size: int,
        n_epochs: int = 1,
        metrics: Optional[
            Tuple[Callable[[jnp.ndarray, jnp.ndarray, Array], Array], ...]
        ] = None,
        val_data_loader: Optional[DataLoader] = None,
        val_dataset_size: Optional[int] = None,
        verbose: bool = True,
    ) -> Tuple[CalibState, Status]:
        training_losses_and_metrics = collections.defaultdict(list)
        val_losses_and_metrics = collections.defaultdict(list)

        state, data_loaders, outputs_loaders, rng = self.on_train_start(
            state,
            [training_data_loader, val_data_loader],
            [self._calib_outputs_loader, self._val_outputs_loader],
            rng,
        )
        training_data_loader, val_data_loader = data_loaders
        calib_outputs_loader, val_outputs_loader = outputs_loaders

        progress_bar = trange(n_epochs, desc="Epoch")
        for epoch in progress_bar:
            # training loop
            (
                state,
                training_losses_and_metrics_current_epoch,
                training_batch_metrics_str,
            ) = self._training_loop(
                epoch,
                fun,
                metrics,
                rng,
                state,
                training_data_loader,
                calib_outputs_loader,
                training_dataset_size,
                verbose,
                progress_bar,
            )
            # keep track of training losses and metrics [granularity=epoch]
            for k in training_losses_and_metrics_current_epoch.keys():
                training_losses_and_metrics[k].append(
                    training_losses_and_metrics_current_epoch[k]
                )

            # validation loop
            if self.should_perform_validation(val_data_loader, epoch):
                # performance evaluation on the whole validation dataset
                state = self.on_val_start(state)
                (
                    val_losses_and_metrics_current_epoch,
                    val_epoch_metrics_str,
                ) = self._val_loop(
                    fun=fun,
                    metrics=metrics,
                    rng=rng,
                    state=state,
                    val_data_loader=val_data_loader,
                    val_outputs_loader=val_outputs_loader,
                    val_dataset_size=val_dataset_size,
                    verbose=verbose,
                )
                if verbose:
                    logging.info(f"Epoch: {epoch + 1} | " + val_epoch_metrics_str)
                # keep track of training losses and metrics [granularity=epoch] and check for early stopping
                for k in val_losses_and_metrics_current_epoch.keys():
                    val_losses_and_metrics[k].append(
                        val_losses_and_metrics_current_epoch[k]
                    )
                # check for early stopping
                if self.is_early_stopping_active and self.early_stopping.should_stop:
                    logging.info("[Early Stopping] Stopping training...")
                    break

        # aggregate
        training_status = {
            k: jnp.array(v) for k, v in training_losses_and_metrics.items()
        }
        val_status = {k: jnp.array(v) for k, v in val_losses_and_metrics.items()}
        status = dict(**training_status, **val_status)

        state = self.on_train_end(state)
        return state, status

    def _training_loop(
        self,
        current_epoch: int,
        fun: Callable,
        metrics: Optional[
            Tuple[Callable[[jnp.ndarray, jnp.ndarray, Array], Array], ...]
        ],
        rng: PRNGKeyArray,
        state: CalibState,
        training_data_loader: DataLoader,
        calib_outputs_loader: TargetsLoader,
        training_dataset_size: int,
        verbose: bool,
        progress_bar: TqdmDecorator,
    ) -> Tuple[CalibState, Dict[str, float], str]:
        training_losses_and_metrics_epoch_all_steps = []
        training_batch_metrics_str = ""
        for step, (batch, outputs) in enumerate(
            zip(training_data_loader, calib_outputs_loader)
        ):
            # forward and backward pass
            state, aux = self.training_step(
                state, batch, outputs, fun, rng, training_dataset_size
            )
            # compute training losses and metrics for the current batch
            training_losses_and_metrics_current_batch = self.training_step_end(
                current_epoch=current_epoch,
                state=state,
                aux=aux,
                batch=batch,
                metrics=metrics,
            )
            # keep track of training losses and metrics [granularity=batch]
            training_losses_and_metrics_epoch_all_steps.append(
                training_losses_and_metrics_current_batch
            )
            # logging
            if verbose:
                training_batch_metrics_str = " | ".join(
                    [
                        f"{m}: {round(float(v), 5)}"
                        for m, v in training_losses_and_metrics_current_batch.items()
                    ]
                )
                progress_bar.set_description(
                    f"Epoch: {current_epoch + 1} | " + training_batch_metrics_str,
                    refresh=True,
                )

        # compute training losses and metrics avg for the current epoch + other ops (if needed)
        training_losses_and_metrics_current_epoch = self.training_epoch_end(
            training_losses_and_metrics_epoch_all_steps
        )

        return (
            state,
            training_losses_and_metrics_current_epoch,
            training_batch_metrics_str,
        )

    def training_step(
        self,
        state: CalibState,
        batch: Batch,
        outputs: Array,
        fun: Callable,
        rng: PRNGKeyArray,
        n_data: int,
    ) -> Tuple[CalibState, Dict[str, Any]]:
        # ensure to use a different key at each step
        model_key = random.fold_in(rng, state.step)

        grad_fn = value_and_grad(
            lambda params: self.training_loss_step(
                fun, params, batch, outputs, state.mutable, model_key, n_data
            ),
            has_aux=True,
        )
        (loss, aux), grad = grad_fn(state.params)
        grad, loss = self.sync_gradients_and_loss(grad, loss)

        state = state.apply_gradients(grads=grad, mutable=aux["mutable"])
        return (
            state,
            {
                "loss": loss,
                "outputs": aux["outputs"],
                "logging_kwargs": aux["logging_kwargs"],
            },
        )

    @abc.abstractmethod
    def training_loss_step(
        self,
        fun: Callable[[Any], Union[float, Tuple[float, dict]]],
        params: CalibParams,
        batch: Batch,
        outputs: Array,
        mutable: CalibMutable,
        rng: PRNGKeyArray,
        n_data: int,
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        pass

    def training_step_end(
        self,
        current_epoch: int,
        state: CalibState,
        aux: Dict[str, Any],
        batch: Batch,
        metrics: Optional[
            Tuple[Callable[[jnp.ndarray, jnp.ndarray, Array], Array], ...]
        ],
    ) -> Dict[str, jnp.ndarray]:
        if (
            self.save_checkpoint_dir
            and self.save_every_n_steps
            and current_epoch % self.save_every_n_steps == 0
        ):
            self.save_checkpoint(
                state, self.save_checkpoint_dir, keep=self.keep_top_n_checkpoints
            )
        training_losses_and_metrics = {"loss": aux["loss"]}

        if aux["logging_kwargs"] is not None:
            for k, v in aux["logging_kwargs"].items():
                training_losses_and_metrics[k] = v

        if not self.disable_training_metrics_computation and metrics is not None:
            preds = self.predict_fn(aux["outputs"])
            uncertainties = self.uncertainty_fn(aux["outputs"])
            if self.multi_gpu:
                training_batch_metrics = self.compute_metrics(
                    preds.reshape((preds.shape[0] * preds.shape[1],) + preds.shape[2:]),
                    uncertainties.reshape(
                        (uncertainties.shape[0] * uncertainties.shape[1],)
                        + uncertainties.shape[2:]
                    ),
                    batch[1].reshape(
                        (batch[1].shape[0] * batch[1].shape[1],) + batch[1].shape[2:]
                    ),
                    metrics,
                )
            else:
                training_batch_metrics = self.compute_metrics(
                    preds, uncertainties, batch[1], metrics
                )
            for k, v in training_batch_metrics.items():
                training_losses_and_metrics[k] = v
        return training_losses_and_metrics

    def _val_loop(
        self,
        fun: Callable,
        metrics: Optional[
            Tuple[Callable[[jnp.ndarray, jnp.ndarray, Array], Array], ...]
        ],
        rng: PRNGKeyArray,
        state: CalibState,
        val_data_loader: DataLoader,
        val_outputs_loader: TargetsLoader,
        val_dataset_size: int,
        verbose: bool = True,
    ) -> Tuple[Dict[str, float], str]:
        val_losses_and_metrics_epoch_all_steps = []
        val_epoch_metrics_str = ""
        for batch, outputs in zip(val_data_loader, val_outputs_loader):
            val_losses_and_metrics_current_batch = self.val_step(
                state, batch, outputs, fun, rng, val_dataset_size, metrics,
            )
            val_losses_and_metrics_epoch_all_steps.append(
                val_losses_and_metrics_current_batch
            )
        # compute validation losses and metrics for the current epoch
        val_losses_and_metrics_current_epoch = self.val_epoch_end(
            val_losses_and_metrics_epoch_all_steps, state
        )
        # logging
        if verbose:
            val_epoch_metrics_str = " | ".join(
                [
                    f"{m}: {round(float(v), 5)}"
                    for m, v in val_losses_and_metrics_current_epoch.items()
                ]
            )
        return val_losses_and_metrics_current_epoch, val_epoch_metrics_str

    def val_step(
        self,
        state: CalibState,
        batch: Batch,
        outputs: Array,
        fun: Callable,
        rng: PRNGKeyArray,
        n_data: int,
        metrics: Optional[
            Tuple[Callable[[jnp.ndarray, jnp.ndarray, Array], Array], ...]
        ] = None,
    ) -> Dict[str, jnp.ndarray]:
        val_loss, aux = self.val_loss_step(state, batch, outputs, fun, rng, n_data)
        val_metrics = self.val_metrics_step(aux, batch, metrics)
        return {"val_loss": val_loss, **val_metrics}

    @abc.abstractmethod
    def val_loss_step(
        self,
        state: CalibState,
        batch: Batch,
        outputs: Array,
        fun: Callable,
        rng: PRNGKeyArray,
        n_data: int,
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        pass

    def val_metrics_step(
        self,
        aux: Dict[str, jnp.ndarray],
        batch: Batch,
        metrics: Optional[
            Tuple[Callable[[jnp.ndarray, jnp.ndarray, Array], Array], ...]
        ] = None,
    ) -> Dict[str, jnp.ndarray]:
        if metrics is not None:
            val_metrics = self.compute_metrics(
                self.predict_fn(aux["outputs"]),
                self.uncertainty_fn(aux["outputs"]),
                batch[1],
                metrics,
            )
            return {f"val_{m}": v for m, v in val_metrics.items()}
        else:
            return {}

    def training_epoch_end(
        self, training_losses_and_metrics_current_epoch: List[Dict[str, jnp.ndarray]]
    ) -> Dict[str, float]:
        return self._get_mean_losses_and_metrics(
            training_losses_and_metrics_current_epoch
        )

    def val_epoch_end(
        self,
        val_losses_and_metrics_current_epoch: List[Dict[str, jnp.ndarray]],
        state: CalibState,
    ) -> Dict[str, float]:
        val_losses_and_metrics_current_epoch = self._get_mean_losses_and_metrics(
            val_losses_and_metrics_current_epoch
        )
        # early stopping
        improved = self.early_stopping_update(val_losses_and_metrics_current_epoch)
        if improved and self.save_checkpoint_dir:
            self.save_checkpoint(state, self.save_checkpoint_dir, force_save=True)
        return val_losses_and_metrics_current_epoch

    def _get_mean_losses_and_metrics(
        self, losses_and_metrics: List[Dict[str, jnp.ndarray]]
    ) -> Dict[str, float]:
        losses_and_metrics = stack_forest(losses_and_metrics)
        losses_and_metrics = tree_map(lambda x: x.mean(), losses_and_metrics)
        return losses_and_metrics

    def should_perform_validation(
        self, val_data_loader: Optional[DataLoader], epoch: int
    ) -> bool:
        return (
            val_data_loader is not None
            and self.eval_every_n_epochs > 0
            and epoch % self.eval_every_n_epochs == 0
        )

    @staticmethod
    def sync_gradients_and_loss(
        grad: jnp.ndarray, loss: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return grad, loss

    def on_train_start(
        self,
        state: CalibState,
        data_loaders: List[DataLoader],
        outputs_loaders: List[TargetsLoader],
        rng: PRNGKeyArray,
    ) -> Tuple[CalibState, List[DataLoader], List[TargetsLoader], PRNGKeyArray]:
        return state, data_loaders, outputs_loaders, rng

    def on_train_end(self, state: CalibState) -> CalibState:
        self.save_checkpoint(
            state,
            save_checkpoint_dir=self.save_checkpoint_dir,
            keep=self.keep_top_n_checkpoints,
            force_save=True,
        )
        return state

    def on_val_start(self, state: CalibState) -> CalibState:
        return state

    def compute_metrics(
        self,
        preds: Array,
        uncertainties: Array,
        targets: Array,
        metrics: Optional[
            Tuple[Callable[[jnp.ndarray, jnp.ndarray, Array], Array], ...]
        ],
    ) -> Dict[str, Array]:
        metrics_vals = {}
        for metric in metrics:
            metrics_vals[metric.__name__] = metric(preds, uncertainties, targets)
        return metrics_vals


class JittedMixin:
    @partial(jax.jit, static_argnums=(0, 4, 6))
    def training_step(
        self,
        state: CalibState,
        batch: Batch,
        outputs: Array,
        fun: Callable,
        rng: PRNGKeyArray,
        n_data: int,
    ) -> Tuple[CalibState, Dict[str, Any]]:
        return super().training_step(state, batch, outputs, fun, rng, n_data)

    @partial(jax.jit, static_argnums=(0, 4, 6))
    def val_loss_step(
        self,
        state: CalibState,
        batch: Batch,
        outputs: Array,
        fun: Callable,
        rng: PRNGKeyArray,
        n_data: int,
    ) -> Dict[str, jnp.ndarray]:
        return super().val_loss_step(state, batch, outputs, fun, rng, n_data)


class MultiGPUMixin:
    all_reduce_mean = jax.pmap(lambda x: lax.pmean(x, "x"), "x")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multi_gpu = True

    @staticmethod
    def _add_device_dim_to_data_loader(data_loader: DataLoader) -> DataLoader:
        def _reshape_batch(batch):
            n_devices = jax.local_device_count()
            if batch.shape[0] % n_devices != 0:
                raise ValueError(
                    f"The size of all batches must be a multiple of {n_devices}, that is the number of "
                    f"available devices. However, a batch with shape {batch.shape[0]} was found. "
                    f"Please set an appropriate batch size."
                )
            return batch.reshape((n_devices, -1) + batch.shape[1:])

        class DataLoaderWrapper:
            def __init__(self, data_loader: DataLoader):
                self._data_loader = data_loader

            def __iter__(self):
                data_loader = map(
                    lambda batch: tree_map(_reshape_batch, batch), self._data_loader
                )
                data_loader = jax_utils.prefetch_to_device(data_loader, 2)
                yield from data_loader

        return (
            DataLoaderWrapper(data_loader) if data_loader is not None else data_loader
        )

    @staticmethod
    def _add_device_dim_to_outputs_loader(
        outputs_loader: TargetsLoader,
    ) -> TargetsLoader:
        def _reshape_batch(batch):
            n_devices = jax.local_device_count()
            if batch.shape[0] % n_devices != 0:
                raise ValueError(
                    f"The size of all output batches must be a multiple of {n_devices}, that is the number of "
                    f"available devices. However, a batch of outputs with shape {batch.shape[0]} was found. "
                    f"Please set an appropriate batch size."
                )
            return batch.reshape((n_devices, -1) + batch.shape[1:])

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

    @staticmethod
    def sync_mutable(state: CalibState) -> CalibState:
        return (
            state.replace(mutable=MultiGPUMixin.all_reduce_mean(state.mutable))
            if state.mutable["output_calibrator"] is not None
            else state
        )

    @staticmethod
    def sync_gradients_and_loss(
        grads: jnp.ndarray, loss: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        grad = lax.pmean(grads, axis_name="batch")
        loss = lax.pmean(loss, axis_name="batch")
        return grad, loss

    def save_checkpoint(
        self,
        state: CalibState,
        save_checkpoint_dir: Path,
        keep: int = 1,
        force_save: bool = False,
        prefix: str = "checkpoint_",
    ) -> None:
        state = self.sync_mutable(state)
        state = jax.device_get(tree_map(lambda x: x[0], state))
        return super(MultiGPUMixin, self).save_checkpoint(
            state, save_checkpoint_dir, keep, force_save, prefix
        )

    def on_train_start(
        self,
        state: CalibState,
        data_loaders: List[DataLoader],
        outputs_loaders: List[TargetsLoader],
        rng: PRNGKeyArray,
    ) -> Tuple[CalibState, List[DataLoader], List[TargetsLoader], PRNGKeyArray]:
        state, data_loaders, outputs_loaders, rng = super(
            MultiGPUMixin, self
        ).on_train_start(state, data_loaders, outputs_loaders, rng)
        state = jax_utils.replicate(state)
        data_loaders = [
            self._add_device_dim_to_data_loader(dl) if dl is not None else dl
            for dl in data_loaders
        ]
        outputs_loaders = [
            self._add_device_dim_to_outputs_loader(ol) if ol is not None else ol
            for ol in outputs_loaders
        ]
        model_key = random.split(rng, jax.local_device_count())
        return state, data_loaders, outputs_loaders, model_key

    def on_train_end(self, state: CalibState) -> CalibState:
        state = super(MultiGPUMixin, self).on_train_end(state)
        return jax.device_get(tree_map(lambda x: x[0], state))

    @partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(0, 4, 6))
    def training_step(
        self,
        state: CalibState,
        batch: Batch,
        outputs: Array,
        fun: Callable,
        rng: PRNGKeyArray,
        n_data: int,
    ) -> Tuple[CalibState, Dict[str, Any]]:
        return super().training_step(state, batch, outputs, fun, rng, n_data)

    def training_step_end(
        self,
        current_epoch: int,
        state: CalibState,
        aux: Dict[str, Any],
        batch: Batch,
        metrics: Optional[Tuple[Callable[[jnp.ndarray, Array], float], ...]],
    ) -> Dict[str, jnp.ndarray]:
        training_losses_and_metrics = super(MultiGPUMixin, self).training_step_end(
            current_epoch, state, aux, batch, metrics
        )
        return tree_map(lambda x: x.mean(), training_losses_and_metrics)

    def on_val_start(self, state: CalibState) -> CalibState:
        state = super(MultiGPUMixin, self).on_val_start(state)
        if state.mutable["output_calibrator"] is not None:
            state = self.sync_mutable(state)
        return state

    @partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(0, 4, 6))
    def val_loss_step(
        self,
        state: CalibState,
        batch: Batch,
        outputs: Array,
        fun: Callable,
        rng: PRNGKeyArray,
        n_data: int,
    ) -> Dict[str, jnp.ndarray]:
        val_losses = super().val_loss_step(state, batch, outputs, fun, rng, n_data)
        return lax.pmean(val_losses, axis_name="batch")

    def val_metrics_step(
        self,
        aux: Dict[str, jnp.ndarray],
        batch: Batch,
        metrics: Optional[
            Tuple[Callable[[jnp.ndarray, jnp.ndarray, Array], Array], ...]
        ] = None,
    ) -> Dict[str, jnp.ndarray]:
        outputs = aux["outputs"]
        outputs = outputs.reshape(outputs.shape[0] * outputs.shape[1], -1)
        targets = batch[1].reshape(batch[1].shape[0] * batch[1].shape[1], -1)
        if metrics is not None:
            val_metrics = self.compute_metrics(
                self.predict_fn(outputs), self.uncertainty_fn(outputs), targets, metrics
            )
            return {f"val_{m}": v for m, v in val_metrics.items()}
        else:
            return {}
