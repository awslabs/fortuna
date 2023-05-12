import abc
import collections
import logging
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import jax
import jax.numpy as jnp
from flax import jax_utils
from flax.core import FrozenDict
from flax.training.common_utils import stack_forest
from jax import (
    lax,
    random,
    value_and_grad,
)
from jax._src.prng import PRNGKeyArray
from jax.tree_util import tree_map
from optax._src.base import PyTree
from tqdm import trange
from tqdm.std import tqdm as TqdmDecorator

from fortuna.data.loader import DataLoader
from fortuna.training.callback import Callback
from fortuna.training.mixin import (
    InputValidatorMixin,
    WithCheckpointingMixin,
    WithEarlyStoppingMixin,
)
from fortuna.training.train_state import TrainState
from fortuna.typing import (
    Array,
    Batch,
    CalibMutable,
    CalibParams,
    Mutable,
    Params,
    Path,
    Status,
)
from fortuna.utils.builtins import HashableMixin
from fortuna.utils.training import clip_grandients_by_norm


class TrainerABC(
    HashableMixin,
    WithCheckpointingMixin,
    WithEarlyStoppingMixin,
    InputValidatorMixin,
    metaclass=abc.ABCMeta,
):
    def __init__(
        self,
        *args,
        predict_fn: Callable[[jnp.ndarray], jnp.ndarray],
        uncertainty_fn: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
        save_checkpoint_dir: Optional[Path] = None,
        save_every_n_steps: Optional[int] = None,
        keep_top_n_checkpoints: int = 2,
        disable_training_metrics_computation: bool = False,
        eval_every_n_epochs: int = 1,
        **kwargs,
    ):
        super(TrainerABC, self).__init__(*args, **kwargs)
        self.predict_fn = predict_fn
        self.uncertainty_fn = uncertainty_fn
        self.save_checkpoint_dir = save_checkpoint_dir
        self.save_every_n_steps = save_every_n_steps
        self.keep_top_n_checkpoints = keep_top_n_checkpoints
        self.disable_training_metrics_computation = disable_training_metrics_computation
        self.eval_every_n_epochs = eval_every_n_epochs
        self._global_training_step = 0
        self._unravel = None
        self.multi_device = False

    @abc.abstractmethod
    def training_loss_step(
        self,
        loss_fun: Callable[[Any], Union[float, Tuple[float, dict]]],
        params: Params,
        batch: Batch,
        mutable: Mutable,
        rng: PRNGKeyArray,
        n_data: int,
        unravel: Optional[Callable[[any], PyTree]] = None,
        calib_params: Optional[CalibParams] = None,
        calib_mutable: Optional[CalibMutable] = None,
        kwargs: FrozenDict[str, Any] = FrozenDict(),
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        pass

    def training_step(
        self,
        state: TrainState,
        batch: Batch,
        loss_fun: Callable,
        rng: PRNGKeyArray,
        n_data: int,
        unravel: Optional[Callable[[any], PyTree]] = None,
        kwargs: FrozenDict[str, Any] = FrozenDict(),
    ) -> Tuple[TrainState, Dict[str, Any]]:
        if state.dynamic_scale is None:
            value_and_grad_fn = value_and_grad
        else:
            value_and_grad_fn = state.dynamic_scale.value_and_grad

        grad_fn = value_and_grad_fn(
            lambda params: self.training_loss_step(
                loss_fun,
                params,
                batch,
                state.mutable,
                rng,
                n_data,
                unravel,
                state.calib_params,
                state.calib_mutable,
                kwargs,
            ),
            has_aux=True,
        )
        outputs = grad_fn(state.params)
        if state.dynamic_scale is None:
            (loss, aux), grad = outputs
            loss = self._sync_array(loss)
        else:
            # dynamic_scale.value_and_grad takes care of averaging gradients and loss across replicas (no need to call sync)
            dynamic_scale, is_fin, aux, grad = outputs
            loss, aux = aux

        max_grad_norm = kwargs.get("max_grad_norm", 0.0) or 0.0
        gradient_accumulation_steps = (
            kwargs.get("gradient_accumulation_steps", 0.0) or 0.0
        )
        if gradient_accumulation_steps > 1:
            new_state = self.params_update_with_grad_accumulation(
                state, grad, aux, gradient_accumulation_steps, max_grad_norm
            )
        else:
            grad = self._sync_array(grad) if state.dynamic_scale is None else grad
            if max_grad_norm > 0:
                grad = clip_grandients_by_norm(grad, max_grad_norm)
            new_state = state.apply_gradients(grads=grad, mutable=aux["mutable"])

        if new_state.dynamic_scale is not None:
            # if is_fin is False the gradients contain Inf/NaNs and optimizer state and
            # params should be restored (= skip this step).
            new_state = new_state.replace(
                opt_state=jax.tree_util.tree_map(
                    partial(jnp.where, is_fin), new_state.opt_state, state.opt_state
                ),
                params=jax.tree_util.tree_map(
                    partial(jnp.where, is_fin), new_state.params, state.params
                ),
                dynamic_scale=dynamic_scale,
            )
        return (
            new_state,
            {
                "loss": loss,
                "outputs": aux["outputs"],
                "logging_kwargs": aux["logging_kwargs"],
            },
        )

    def params_update_with_grad_accumulation(
        self,
        state: TrainState,
        grad: jnp.ndarray,
        aux: Dict[str, Any],
        gradient_accumulation_steps: int,
        max_grad_norm: float,
    ) -> TrainState:
        if state.grad_accumulated is not None:
            grad_acc = tree_map(lambda x, y: x + y, grad, state.grad_accumulated)
        else:
            # grad_accumulated is not initialized yet, will do no now
            grad_acc = grad

        def update_fn(state):
            grad = tree_map(lambda x: x / gradient_accumulation_steps, grad_acc)
            grad = self._sync_array(grad) if state.dynamic_scale is None else grad
            if max_grad_norm > 0:
                grad = clip_grandients_by_norm(grad, max_grad_norm)
            return state.apply_gradients(
                grads=grad,
                grad_accumulated=jax.tree_map(jnp.zeros_like, grad),
                mutable=aux["mutable"],
            )

        new_state = jax.lax.cond(
            (state.step + 1) % gradient_accumulation_steps == 0,
            lambda state: update_fn(state),
            lambda state: state.replace(
                grad_accumulated=grad_acc, step=state.step + 1, mutable=aux["mutable"]
            ),
            state,
        )
        return new_state

    @abc.abstractmethod
    def validation_step(
        self,
        state: TrainState,
        batch: Batch,
        loss_fun: Callable,
        rng: PRNGKeyArray,
        n_data: int,
        metrics: Optional[Tuple[Callable[[jnp.ndarray, Array], float], ...]] = None,
        unravel: Optional[Callable[[any], PyTree]] = None,
        kwargs: FrozenDict[str, Any] = FrozenDict(),
    ) -> Dict[str, jnp.ndarray]:
        pass

    def training_step_end(
        self,
        current_epoch: int,
        state: TrainState,
        aux: Dict[str, Any],
        batch: Batch,
        metrics: Optional[
            Tuple[
                Union[
                    Callable[[jnp.ndarray, Array], float],
                    Callable[[jnp.ndarray, jnp.ndarray, Array], float],
                ],
                ...,
            ]
        ],
        callbacks: Optional[List[Callback]] = None,
        kwargs: FrozenDict[str, Any] = FrozenDict(),
    ) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
        if (
            self.save_checkpoint_dir is not None
            and self.save_every_n_steps is not None
            and self.save_every_n_steps > 0
            and self._global_training_step >= self.save_every_n_steps
            and self._global_training_step % self.save_every_n_steps == 0
        ):
            self.save_checkpoint(
                state, self.save_checkpoint_dir, keep=self.keep_top_n_checkpoints
            )
        training_losses_and_metrics = {"loss": aux["loss"]}

        if aux["logging_kwargs"] is not None:
            for k, v in aux["logging_kwargs"].items():
                training_losses_and_metrics[k] = v

        if not self.disable_training_metrics_computation and metrics is not None:
            preds = self.predict_fn(aux["outputs"], train=True)
            if self.uncertainty_fn is not None:
                uncertainties = self.uncertainty_fn(aux["outputs"])
                if self.multi_device:
                    training_batch_metrics = self.compute_metrics(
                        preds.reshape(
                            (preds.shape[0] * preds.shape[1],) + preds.shape[2:]
                        ),
                        batch[1].reshape(
                            (batch[1].shape[0] * batch[1].shape[1],)
                            + batch[1].shape[2:]
                        ),
                        metrics,
                        uncertainties.reshape(
                            (uncertainties.shape[0] * uncertainties.shape[1],)
                            + uncertainties.shape[2:]
                        ),
                    )
                else:
                    training_batch_metrics = self.compute_metrics(
                        preds, batch[1], metrics, uncertainties
                    )
            else:
                if self.multi_device:
                    training_batch_metrics = self.compute_metrics(
                        preds=preds.reshape(
                            (preds.shape[0] * preds.shape[1],) + preds.shape[2:]
                        ),
                        targets=batch[1].reshape(
                            (batch[1].shape[0] * batch[1].shape[1],)
                            + batch[1].shape[2:]
                        ),
                        metrics=metrics,
                    )
                else:
                    training_batch_metrics = self.compute_metrics(
                        preds, batch[1], metrics
                    )
            for k, v in training_batch_metrics.items():
                training_losses_and_metrics[k] = v

        state = self._callback_loop(state, callbacks, "training_step_end")
        return state, training_losses_and_metrics

    def training_epoch_start(
        self, state: TrainState, callbacks: Optional[List[Callback]] = None
    ) -> TrainState:
        return self._callback_loop(state, callbacks, "training_epoch_start")

    def training_epoch_end(
        self,
        training_losses_and_metrics_current_epoch: List[Dict[str, jnp.ndarray]],
        state: TrainState,
        callbacks: Optional[List[Callback]] = None,
    ) -> Tuple[TrainState, Dict[str, float]]:
        mean_losses_and_metrics = self._get_mean_losses_and_metrics(
            training_losses_and_metrics_current_epoch
        )
        state = self._callback_loop(state, callbacks, "training_epoch_end")
        return state, mean_losses_and_metrics

    def validation_epoch_end(
        self,
        validation_losses_and_metrics_current_epoch: List[Dict[str, jnp.ndarray]],
        state: TrainState,
    ) -> Dict[str, float]:
        validation_losses_and_metrics_current_epoch = self._get_mean_losses_and_metrics(
            validation_losses_and_metrics_current_epoch
        )
        # early stopping
        improved = self.early_stopping_update(
            validation_losses_and_metrics_current_epoch
        )
        if improved and self.save_checkpoint_dir:
            self.save_checkpoint(state, self.save_checkpoint_dir, force_save=True)
        return validation_losses_and_metrics_current_epoch

    def train(
        self,
        rng: PRNGKeyArray,
        state: TrainState,
        loss_fun: Callable,
        training_dataloader: DataLoader,
        training_dataset_size: int,
        n_epochs: int = 1,
        metrics: Optional[Tuple[Callable[[jnp.ndarray, Array], float], ...]] = None,
        validation_dataloader: Optional[DataLoader] = None,
        validation_dataset_size: Optional[int] = None,
        verbose: bool = True,
        unravel: Optional[Callable[[any], PyTree]] = None,
        callbacks: Optional[List[Callback]] = None,
        **kwargs,
    ) -> Tuple[TrainState, Status]:
        training_kwargs = FrozenDict(kwargs)
        if validation_dataloader:
            assert (
                validation_dataset_size is not None
            ), "`validation_dataset_size` is required when `validation_dataloader` is provided."

        training_losses_and_metrics = collections.defaultdict(list)
        validation_losses_and_metrics = collections.defaultdict(list)

        state, dataloaders, rng = self.on_train_start(
            state, [training_dataloader, validation_dataloader], rng
        )
        training_dataloader, validation_dataloader = dataloaders

        progress_bar = trange(n_epochs, desc="Epoch")
        for epoch in progress_bar:
            # training loop
            (
                state,
                training_losses_and_metrics_current_epoch,
                training_batch_metrics_str,
            ) = self._training_loop(
                epoch,
                loss_fun,
                metrics,
                rng,
                state,
                training_dataloader,
                training_dataset_size,
                training_kwargs,
                verbose,
                progress_bar,
                unravel=unravel,
                callbacks=callbacks,
            )
            # keep track of training losses and metrics [granularity=epoch]
            for k in training_losses_and_metrics_current_epoch.keys():
                training_losses_and_metrics[k].append(
                    training_losses_and_metrics_current_epoch[k]
                )

            # validation loop
            if self.should_perform_validation(validation_dataloader, epoch):
                # performance evaluation on the whole validation dataset
                state = self.on_validation_start(state)
                (
                    validation_losses_and_metrics_current_epoch,
                    validation_epoch_metrics_str,
                ) = self._validation_loop(
                    loss_fun=loss_fun,
                    metrics=metrics,
                    rng=rng,
                    state=state,
                    training_kwargs=training_kwargs,
                    validation_dataloader=validation_dataloader,
                    validation_dataset_size=validation_dataset_size,
                    verbose=verbose,
                    unravel=unravel,
                )
                if verbose:
                    logging.info(
                        f"Epoch: {epoch + 1} | " + validation_epoch_metrics_str
                    )
                # keep track of training losses and metrics [granularity=epoch] and check for early stopping
                for k in validation_losses_and_metrics_current_epoch.keys():
                    validation_losses_and_metrics[k].append(
                        validation_losses_and_metrics_current_epoch[k]
                    )
                # check for early stopping
                if self.is_early_stopping_active and self._early_stopping.should_stop:
                    logging.info("[Early Stopping] Stopping training...")
                    break

        # aggregate
        training_status = {
            k: jnp.array(v) for k, v in training_losses_and_metrics.items()
        }
        validation_status = {
            k: jnp.array(v) for k, v in validation_losses_and_metrics.items()
        }
        status = dict(**training_status, **validation_status)

        state = self.on_train_end(state)
        return state, status

    def _training_loop(
        self,
        current_epoch: int,
        loss_fun: Callable,
        metrics: Optional[Tuple[Callable[[jnp.ndarray, Array], jnp.ndarray], ...]],
        rng: PRNGKeyArray,
        state: TrainState,
        training_dataloader: DataLoader,
        training_dataset_size: int,
        training_kwargs: FrozenDict[str, Any],
        verbose: bool,
        progress_bar: TqdmDecorator,
        unravel: Optional[Callable[[any], PyTree]] = None,
        callbacks: Optional[List[Callback]] = None,
    ) -> Tuple[TrainState, Dict[str, float], str]:
        gradient_accumulation_steps = (
            training_kwargs.get("gradient_accumulation_steps", 0) or 0
        )
        training_losses_and_metrics_epoch_all_steps = []
        training_batch_metrics_str = ""
        state = self.training_epoch_start(state, callbacks)
        # ensure to use a different key at each step
        model_key = self.training_step_start(rng, state.step)
        for step, batch in enumerate(training_dataloader):
            # forward and backward pass
            state, aux = self.training_step(
                state,
                batch,
                loss_fun,
                model_key,
                training_dataset_size,
                unravel,
                training_kwargs,
            )
            self._global_training_step += 1
            if (gradient_accumulation_steps > 0) and (
                (step + 1) % gradient_accumulation_steps != 0
            ):
                continue
            # update model key
            model_key = self.training_step_start(rng, state.step)
            # compute training losses and metrics for the current batch
            state, training_losses_and_metrics_current_batch = self.training_step_end(
                current_epoch=current_epoch,
                state=state,
                aux=aux,
                batch=batch,
                metrics=metrics,
                callbacks=callbacks,
                kwargs=training_kwargs,
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
        state, training_losses_and_metrics_current_epoch = self.training_epoch_end(
            training_losses_and_metrics_epoch_all_steps, state, callbacks
        )

        return (
            state,
            training_losses_and_metrics_current_epoch,
            training_batch_metrics_str,
        )

    def _validation_loop(
        self,
        loss_fun: Callable,
        metrics: Optional[Tuple[Callable[[jnp.ndarray, Array], float], ...]],
        rng: PRNGKeyArray,
        state: TrainState,
        training_kwargs: FrozenDict[str, Any],
        validation_dataloader: DataLoader,
        validation_dataset_size: int,
        verbose: bool = True,
        unravel: Optional[Callable[[any], PyTree]] = None,
    ) -> Tuple[Dict[str, float], str]:
        validation_losses_and_metrics_epoch_all_steps = []
        validation_epoch_metrics_str = ""
        for batch in validation_dataloader:
            validation_losses_and_metrics_current_batch = self.validation_step(
                state,
                batch,
                loss_fun,
                rng,
                validation_dataset_size,
                metrics,
                unravel,
                training_kwargs,
            )
            validation_losses_and_metrics_epoch_all_steps.append(
                validation_losses_and_metrics_current_batch
            )
        # compute validation losses and metrics for the current epoch
        validation_losses_and_metrics_current_epoch = self.validation_epoch_end(
            validation_losses_and_metrics_epoch_all_steps, state
        )
        # logging
        if verbose:
            validation_epoch_metrics_str = " | ".join(
                [
                    f"{m}: {round(float(v), 5)}"
                    for m, v in validation_losses_and_metrics_current_epoch.items()
                ]
            )
        return validation_losses_and_metrics_current_epoch, validation_epoch_metrics_str

    def _callback_loop(
        self, state: TrainState, callbacks: Optional[List[Callback]], method_name: str
    ) -> TrainState:
        callbacks = callbacks or []
        for callback in callbacks:
            state = getattr(callback, method_name)(state)
        return state

    def _get_mean_losses_and_metrics(
        self, losses_and_metrics: List[Dict[str, jnp.ndarray]]
    ) -> Dict[str, float]:
        losses_and_metrics = stack_forest(losses_and_metrics)
        losses_and_metrics = tree_map(lambda x: x.mean(), losses_and_metrics)
        return losses_and_metrics

    def should_perform_validation(
        self, validation_dataloader: Optional[DataLoader], epoch: int
    ) -> bool:
        return (
            validation_dataloader is not None
            and self.eval_every_n_epochs > 0
            and epoch % self.eval_every_n_epochs == 0
        )

    @staticmethod
    def _sync_array(arr: jnp.ndarray) -> jnp.ndarray:
        return arr

    def on_train_start(
        self,
        state: TrainState,
        dataloaders: List[DataLoader],
        rng: PRNGKeyArray,
    ) -> Tuple[TrainState, List[DataLoader], PRNGKeyArray]:
        return state, dataloaders, rng

    def on_train_end(self, state: TrainState) -> TrainState:
        self.save_checkpoint(
            state,
            save_checkpoint_dir=self.save_checkpoint_dir,
            keep=self.keep_top_n_checkpoints,
            force_save=True,
        )
        return state

    def on_validation_start(self, state: TrainState) -> TrainState:
        return state

    def compute_metrics(
        self,
        preds: Array,
        targets: Array,
        metrics: Optional[Tuple[Callable[[jnp.ndarray, Array], float], ...]],
        uncertainties: Optional[Array] = None,
    ) -> Dict[str, float]:
        metrics_vals = {}
        for metric in metrics:
            metrics_vals[metric.__name__] = (
                metric(preds, targets)
                if uncertainties is None
                else metric(preds, uncertainties, targets)
            )
        return metrics_vals

    def training_step_start(
        self, rng: PRNGKeyArray, step: Union[int, jax.Array]
    ) -> PRNGKeyArray:
        step = step if isinstance(step, int) or step.ndim == 0 else step[0]
        return random.fold_in(rng, step)


class JittedMixin:
    @partial(jax.jit, static_argnums=(0, 3, 5, 6, 7))
    def training_step(
        self,
        state: TrainState,
        batch: Batch,
        loss_fun: Callable,
        rng: PRNGKeyArray,
        n_data: int,
        unravel: Optional[Callable[[any], PyTree]] = None,
        kwargs: FrozenDict[str, Any] = FrozenDict(),
    ) -> Tuple[TrainState, Dict[str, Any]]:
        return super().training_step(
            state, batch, loss_fun, rng, n_data, unravel, kwargs
        )

    @partial(jax.jit, static_argnums=(0, 3, 5, 6, 7, 8))
    def validation_step(
        self,
        state: TrainState,
        batch: Batch,
        loss_fun: Callable,
        rng: PRNGKeyArray,
        n_data: int,
        metrics: Optional[Tuple[Callable[[jnp.ndarray, Array], float], ...]] = None,
        unravel: Optional[Callable[[any], PyTree]] = None,
        kwargs: FrozenDict[str, Any] = FrozenDict(),
    ) -> Dict[str, jnp.ndarray]:
        return super().validation_step(
            state, batch, loss_fun, rng, n_data, metrics, unravel, kwargs
        )


class MultiDeviceMixin:
    all_reduce_mean = jax.pmap(lambda x: lax.pmean(x, "x"), "x")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multi_device = True

    @staticmethod
    def _add_device_dim_to_input_dataloader(dataloader: DataLoader) -> DataLoader:
        def _reshape_input_batch(batch):
            n_devices = jax.local_device_count()
            if batch.shape[0] % n_devices != 0:
                raise ValueError(
                    f"The size of all batches must be a multiple of {n_devices}, that is the number of "
                    f"available devices. Please set an appropriate batch size in the data loader."
                )
            single_input_shape = batch.shape[1:]
            # reshape to (local_devices, device_batch_size, *single_input_shape)
            return batch.reshape((n_devices, -1) + single_input_shape)

        class DataLoaderWrapper:
            def __init__(self, dataloader):
                self.dataloader = dataloader

            def __iter__(self):
                dataloader = map(
                    lambda batch: tree_map(_reshape_input_batch, batch), self.dataloader
                )
                dataloader = jax_utils.prefetch_to_device(dataloader, 2)
                yield from dataloader

        return DataLoaderWrapper(dataloader) if dataloader is not None else dataloader

    @staticmethod
    def _sync_mutable(state: TrainState) -> TrainState:
        return (
            state.replace(mutable=MultiDeviceMixin.all_reduce_mean(state.mutable))
            if state.mutable is not None
            else state
        )

    @staticmethod
    def _sync_array(arr: jnp.ndarray) -> jnp.ndarray:
        arr = lax.pmean(arr, axis_name="batch")
        return arr

    def save_checkpoint(
        self,
        state: TrainState,
        save_checkpoint_dir: Path,
        keep: int = 1,
        force_save: bool = False,
        prefix: str = "checkpoint_",
    ) -> None:
        state = self._sync_mutable(state)
        state = jax.device_get(tree_map(lambda x: x[0], state))
        return super(MultiDeviceMixin, self).save_checkpoint(
            state, save_checkpoint_dir, keep, force_save, prefix
        )

    def on_train_start(
        self, state: TrainState, dataloaders: List[DataLoader], rng: PRNGKeyArray
    ) -> Tuple[TrainState, List[DataLoader], PRNGKeyArray]:
        state, dataloaders, rng = super(MultiDeviceMixin, self).on_train_start(
            state, dataloaders, rng
        )
        state = jax_utils.replicate(state)
        dataloaders = [
            self._add_device_dim_to_input_dataloader(dl) for dl in dataloaders
        ]
        model_key = random.split(rng, jax.local_device_count())
        return state, dataloaders, model_key

    def on_train_end(self, state: TrainState) -> TrainState:
        state = super(MultiDeviceMixin, self).on_train_end(state)
        return jax.device_get(tree_map(lambda x: x[0], state))

    def training_step_start(self, rng: PRNGKeyArray, step: int) -> PRNGKeyArray:
        step = step if isinstance(step, int) or step.ndim == 0 else step[0]
        return jax.vmap(lambda r: random.fold_in(r, step))(rng)

    @partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(0, 3, 5, 6, 7))
    def training_step(
        self,
        state: TrainState,
        batch: Batch,
        loss_fun: Callable,
        rng: PRNGKeyArray,
        n_data: int,
        unravel: Optional[Callable[[any], PyTree]] = None,
        kwargs: FrozenDict[str, Any] = FrozenDict(),
    ) -> Tuple[TrainState, Dict[str, Any]]:
        return super().training_step(
            state, batch, loss_fun, rng, n_data, unravel, kwargs
        )

    def training_step_end(
        self,
        current_epoch: int,
        state: TrainState,
        aux: Dict[str, Any],
        batch: Batch,
        metrics: Optional[Tuple[Callable[[jnp.ndarray, Array], float], ...]],
        callbacks: Optional[List[Callback]] = None,
        kwargs: FrozenDict[str, Any] = FrozenDict(),
    ) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
        state, training_losses_and_metrics = super(
            MultiDeviceMixin, self
        ).training_step_end(
            current_epoch, state, aux, batch, metrics, callbacks, kwargs
        )
        return state, tree_map(lambda x: x.mean(), training_losses_and_metrics)

    def on_validation_start(self, state: TrainState) -> TrainState:
        state = super(MultiDeviceMixin, self).on_validation_start(state)
        state = self._sync_mutable(state)
        return state

    @partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(0, 3, 5, 6, 7, 8))
    def validation_step(
        self,
        state: TrainState,
        batch: Batch,
        loss_fun: Callable,
        rng: PRNGKeyArray,
        n_data: int,
        metrics: Optional[Tuple[Callable[[jnp.ndarray, Array], float], ...]] = None,
        unravel: Optional[Callable[[any], PyTree]] = None,
        kwargs: FrozenDict[str, Any] = FrozenDict(),
    ) -> Dict[str, jnp.ndarray]:
        validation_losses_and_metrics = super().validation_step(
            state, batch, loss_fun, rng, n_data, metrics, unravel, kwargs
        )
        return lax.pmean(validation_losses_and_metrics, axis_name="batch")
