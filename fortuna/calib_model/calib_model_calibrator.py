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
from fortuna.typing import Array, CalibMutable, CalibParams, Path, Status
from fortuna.utils.builtins import HashableMixin


class CalibModelCalibrator(
    HashableMixin,
    WithCheckpointingMixin,
    WithEarlyStoppingMixin,
    InputValidatorMixin,
    metaclass=abc.ABCMeta,
):
    def __init__(
        self,
        *args,
        calib_outputs: Array,
        calib_targets: Array,
        predict_fn: Callable[[jnp.ndarray], jnp.ndarray],
        uncertainty_fn: Callable[[jnp.ndarray], jnp.ndarray],
        val_outputs: Array,
        val_targets: Array,
        save_checkpoint_dir: Optional[Path] = None,
        save_every_n_steps: Optional[int] = None,
        keep_top_n_checkpoints: int = 2,
        disable_training_metrics_computation: bool = False,
        eval_every_n_epochs: int = 1,
        **kwargs,
    ):
        super(CalibModelCalibrator, self).__init__(*args, **kwargs)
        self._calib_outputs = calib_outputs
        self._calib_targets = calib_targets
        self._val_outputs = val_outputs
        self._val_targets = val_targets
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
        n_epochs: int = 1,
        metrics: Optional[
            Tuple[Callable[[jnp.ndarray, jnp.ndarray, Array], Array], ...]
        ] = None,
        verbose: bool = True,
    ) -> Tuple[CalibState, Status]:
        training_losses_and_metrics = collections.defaultdict(list)
        val_losses_and_metrics = collections.defaultdict(list)

        state, targets, outputs, rng = self.on_train_start(
            state,
            [self._calib_targets, self._val_targets],
            [self._calib_outputs, self._val_outputs],
            rng,
        )
        calib_targets, val_targets = targets
        calib_outputs, val_outputs = outputs

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
                calib_targets,
                calib_outputs,
                verbose,
                progress_bar,
            )
            # keep track of training losses and metrics [granularity=epoch]
            for k in training_losses_and_metrics_current_epoch.keys():
                training_losses_and_metrics[k].append(
                    training_losses_and_metrics_current_epoch[k]
                )

            # validation loop
            if self.should_perform_validation(val_targets, epoch):
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
                    targets=val_targets,
                    outputs=val_outputs,
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
        targets: Array,
        outputs: Array,
        verbose: bool,
        progress_bar: TqdmDecorator,
    ) -> Tuple[CalibState, Dict[str, float], str]:
        training_losses_and_metrics_epoch_all_steps = []
        training_batch_metrics_str = ""
        # forward and backward pass
        state, aux = self.training_step(state, targets, outputs, fun, rng)
        # compute training losses and metrics for the current batch
        training_losses_and_metrics_current_batch = self.training_step_end(
            current_epoch=current_epoch,
            state=state,
            aux=aux,
            targets=targets,
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
        targets: Array,
        outputs: Array,
        fun: Callable,
        rng: PRNGKeyArray,
    ) -> Tuple[CalibState, Dict[str, Any]]:
        # ensure to use a different key at each step
        model_key = random.fold_in(rng, state.step)

        grad_fn = value_and_grad(
            lambda params: self.training_loss_step(
                fun, params, targets, outputs, state.mutable, model_key,
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

    def training_loss_step(
        self,
        fun: Callable[[Any], Union[float, Tuple[float, dict]]],
        params: CalibParams,
        targets: Array,
        outputs: Array,
        mutable: CalibMutable,
        rng: PRNGKeyArray,
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        return_aux = ["outputs"]
        if mutable is not None:
            return_aux += ["mutable"]
        loss, aux = fun(
            params=params,
            targets=targets,
            outputs=outputs,
            mutable=mutable,
            rng=rng,
            return_aux=["outputs", "mutable"],
        )
        loss = -loss
        logging_kwargs = None
        return (
            loss,
            {
                "outputs": aux.get("outputs"),
                "mutable": aux.get("mutable"),
                "logging_kwargs": logging_kwargs,
            },
        )

    def training_step_end(
        self,
        current_epoch: int,
        state: CalibState,
        aux: Dict[str, Any],
        targets: Array,
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
                    targets.reshape(
                        (targets.shape[0] * targets.shape[1],) + targets.shape[2:]
                    ),
                    metrics,
                )
            else:
                training_batch_metrics = self.compute_metrics(
                    preds, uncertainties, targets, metrics
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
        targets: Array,
        outputs: Array,
        verbose: bool = True,
    ) -> Tuple[Dict[str, float], str]:
        val_losses_and_metrics_epoch_all_steps = []
        val_epoch_metrics_str = ""
        val_losses_and_metrics_current_batch = self.val_step(
            state, targets, outputs, fun, rng, metrics,
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
        targets: Array,
        outputs: Array,
        fun: Callable,
        rng: PRNGKeyArray,
        metrics: Optional[
            Tuple[Callable[[jnp.ndarray, jnp.ndarray, Array], Array], ...]
        ] = None,
    ) -> Dict[str, jnp.ndarray]:
        val_loss, aux = self.val_loss_step(state, targets, outputs, fun, rng)
        val_metrics = self.val_metrics_step(aux, targets, metrics)
        return {"val_loss": val_loss, **val_metrics}

    def val_loss_step(
        self,
        state: CalibState,
        targets: Array,
        outputs: Array,
        fun: Callable,
        rng: PRNGKeyArray,
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        log_probs, aux = fun(
            params=state.params,
            targets=targets,
            outputs=outputs,
            mutable=state.mutable,
            rng=rng,
            return_aux=["outputs"],
        )
        return -log_probs, aux

    def val_metrics_step(
        self,
        aux: Dict[str, jnp.ndarray],
        targets: Array,
        metrics: Optional[
            Tuple[Callable[[jnp.ndarray, jnp.ndarray, Array], Array], ...]
        ] = None,
    ) -> Dict[str, jnp.ndarray]:
        if metrics is not None:
            val_metrics = self.compute_metrics(
                self.predict_fn(aux["outputs"]),
                self.uncertainty_fn(aux["outputs"]),
                targets,
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
        targets: List[Array],
        outputs: List[Array],
        rng: PRNGKeyArray,
    ) -> Tuple[CalibState, List[Array], List[Array], PRNGKeyArray]:
        return state, targets, outputs, rng

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
    @partial(jax.jit, static_argnums=(0, 4))
    def training_step(
        self,
        state: CalibState,
        targets: Array,
        outputs: Array,
        fun: Callable,
        rng: PRNGKeyArray,
    ) -> Tuple[CalibState, Dict[str, Any]]:
        return super().training_step(state, targets, outputs, fun, rng)

    @partial(jax.jit, static_argnums=(0, 4))
    def val_loss_step(
        self,
        state: CalibState,
        targets: Array,
        outputs: Array,
        fun: Callable,
        rng: PRNGKeyArray,
    ) -> Dict[str, jnp.ndarray]:
        return super().val_loss_step(state, targets, outputs, fun, rng)


class MultiGPUMixin:
    all_reduce_mean = jax.pmap(lambda x: lax.pmean(x, "x"), "x")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multi_gpu = True

    @staticmethod
    def _add_device_dim_to_array(arr: Array) -> Array:
        n_devices = jax.local_device_count()
        if arr.shape[0] % n_devices != 0:
            raise ValueError(
                f"The number of data points of all outputs and targets must be a multiple of {n_devices}, "
                f"that is the number of available devices. However, {arr.shape[0]} were found."
            )
        return arr.reshape((n_devices, -1) + arr.shape[1:]) if arr is not None else arr

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
        targets: List[Array],
        outputs: List[Array],
        rng: PRNGKeyArray,
    ) -> Tuple[CalibState, List[DataLoader], List[TargetsLoader], PRNGKeyArray]:
        state, targets, outputs, rng = super(MultiGPUMixin, self).on_train_start(
            state, targets, outputs, rng
        )
        state = jax_utils.replicate(state)
        targets = [
            self._add_device_dim_to_array(arr) if arr is not None else arr
            for arr in targets
        ]
        outputs = [
            self._add_device_dim_to_array(arr) if arr is not None else arr
            for arr in outputs
        ]
        model_key = random.split(rng, jax.local_device_count())
        return state, targets, outputs, model_key

    def on_train_end(self, state: CalibState) -> CalibState:
        state = super(MultiGPUMixin, self).on_train_end(state)
        return jax.device_get(tree_map(lambda x: x[0], state))

    @partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(0, 4))
    def training_step(
        self,
        state: CalibState,
        targets: Array,
        outputs: Array,
        fun: Callable,
        rng: PRNGKeyArray,
    ) -> Tuple[CalibState, Dict[str, Any]]:
        return super().training_step(state, targets, outputs, fun, rng)

    def training_step_end(
        self,
        current_epoch: int,
        state: CalibState,
        aux: Dict[str, Any],
        targets: Array,
        metrics: Optional[Tuple[Callable[[jnp.ndarray, Array], Array], ...]],
    ) -> Dict[str, jnp.ndarray]:
        training_losses_and_metrics = super(MultiGPUMixin, self).training_step_end(
            current_epoch, state, aux, targets, metrics
        )
        return tree_map(lambda x: x.mean(), training_losses_and_metrics)

    def on_val_start(self, state: CalibState) -> CalibState:
        state = super(MultiGPUMixin, self).on_val_start(state)
        if state.mutable["output_calibrator"] is not None:
            state = self.sync_mutable(state)
        return state

    @partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(0, 4))
    def val_loss_step(
        self,
        state: CalibState,
        targets: Array,
        outputs: Array,
        fun: Callable,
        rng: PRNGKeyArray,
    ) -> Dict[str, jnp.ndarray]:
        val_losses = super().val_loss_step(state, targets, outputs, fun, rng)
        return lax.pmean(val_losses, axis_name="batch")

    def val_metrics_step(
        self,
        aux: Dict[str, jnp.ndarray],
        targets: Array,
        metrics: Optional[
            Tuple[Callable[[jnp.ndarray, jnp.ndarray, Array], Array], ...]
        ] = None,
    ) -> Dict[str, jnp.ndarray]:
        outputs = aux["outputs"]
        outputs = outputs.reshape(outputs.shape[0] * outputs.shape[1], -1)
        targets = targets.reshape(targets.shape[0] * targets.shape[1], -1)
        if metrics is not None:
            val_metrics = self.compute_metrics(
                self.predict_fn(outputs), self.uncertainty_fn(outputs), targets, metrics
            )
            return {f"val_{m}": v for m, v in val_metrics.items()}
        else:
            return {}


class JittedCalibModelCalibrator(JittedMixin, CalibModelCalibrator):
    pass


class MultiGPUCalibModelCalibrator(MultiGPUMixin, CalibModelCalibrator):
    pass
