import abc
import collections
import logging
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from flax.training.common_utils import stack_forest
from jax import (
    random,
    value_and_grad,
)
from jax._src.prng import PRNGKeyArray
import jax.numpy as jnp
from jax.tree_util import tree_map
from tqdm import trange
from tqdm.std import tqdm as TqdmDecorator

from fortuna.data.loader import (
    DataLoader,
    TargetsLoader,
)
from fortuna.output_calib_model.state import OutputCalibState
from fortuna.partitioner.partition_manager.base import PartitionManager
from fortuna.training.mixins.checkpointing import WithCheckpointingMixin
from fortuna.training.mixins.early_stopping import WithEarlyStoppingMixin
from fortuna.training.mixins.input_validator import InputValidatorMixin
from fortuna.typing import (
    Array,
    Batch,
    CalibMutable,
    CalibParams,
    Path,
    Status,
)
from fortuna.utils.builtins import HashableMixin


class OutputCalibratorABC(
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
        partition_manager: PartitionManager,
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
        super(OutputCalibratorABC, self).__init__(
            *args, partition_manager=partition_manager, **kwargs
        )
        self._calib_outputs_loader = calib_outputs_loader
        self._val_outputs_loader = val_outputs_loader
        self.predict_fn = predict_fn
        self.uncertainty_fn = uncertainty_fn
        self.save_checkpoint_dir = save_checkpoint_dir
        self.save_every_n_steps = save_every_n_steps
        self.keep_top_n_checkpoints = keep_top_n_checkpoints
        self.disable_training_metrics_computation = disable_training_metrics_computation
        self.eval_every_n_epochs = eval_every_n_epochs

    def train(
        self,
        rng: PRNGKeyArray,
        state: OutputCalibState,
        loss_fun: Callable,
        training_data_loader: DataLoader,
        training_dataset_size: int,
        n_epochs: int = 1,
        metrics: Optional[
            Tuple[Callable[[jnp.ndarray, jnp.ndarray, Array], Array], ...]
        ] = None,
        val_data_loader: Optional[DataLoader] = None,
        val_dataset_size: Optional[int] = None,
        verbose: bool = True,
    ) -> Tuple[OutputCalibState, Status]:
        training_losses_and_metrics = collections.defaultdict(list)
        validation_losses_and_metrics = collections.defaultdict(list)

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
                loss_fun,
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
                state = self.on_validation_start(state)
                (
                    validation_losses_and_metrics_current_epoch,
                    validation_epoch_metrics_str,
                ) = self._validation_loop(
                    loss_fun=loss_fun,
                    metrics=metrics,
                    rng=rng,
                    state=state,
                    val_data_loader=val_data_loader,
                    val_outputs_loader=val_outputs_loader,
                    val_dataset_size=val_dataset_size,
                    verbose=verbose,
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
        metrics: Optional[
            Tuple[Callable[[jnp.ndarray, jnp.ndarray, Array], Array], ...]
        ],
        rng: PRNGKeyArray,
        state: OutputCalibState,
        training_data_loader: DataLoader,
        calib_outputs_loader: TargetsLoader,
        training_dataset_size: int,
        verbose: bool,
        progress_bar: TqdmDecorator,
    ) -> Tuple[OutputCalibState, Dict[str, float], str]:
        training_losses_and_metrics_epoch_all_steps = []
        training_batch_metrics_str = ""
        for step, (batch, outputs) in enumerate(
            zip(training_data_loader, calib_outputs_loader)
        ):
            # forward and backward pass
            state, aux = self.training_step(
                state, batch, outputs, loss_fun, rng, training_dataset_size
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
        state: OutputCalibState,
        batch: Batch,
        outputs: Array,
        loss_fun: Callable,
        rng: PRNGKeyArray,
        n_data: int,
    ) -> Tuple[OutputCalibState, Dict[str, Any]]:
        # ensure to use a different key at each step
        model_key = random.fold_in(rng, state.step)

        grad_fn = value_and_grad(
            lambda params: self.training_loss_step(
                loss_fun, params, batch, outputs, state.mutable, model_key, n_data
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
        loss_fun: Callable[[Any], Union[float, Tuple[float, dict]]],
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
        state: OutputCalibState,
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
            training_batch_metrics = self.compute_metrics(
                preds, uncertainties, batch[1], metrics
            )
            for k, v in training_batch_metrics.items():
                training_losses_and_metrics[k] = v
        return training_losses_and_metrics

    def _validation_loop(
        self,
        loss_fun: Callable,
        metrics: Optional[
            Tuple[Callable[[jnp.ndarray, jnp.ndarray, Array], Array], ...]
        ],
        rng: PRNGKeyArray,
        state: OutputCalibState,
        val_data_loader: DataLoader,
        val_outputs_loader: TargetsLoader,
        val_dataset_size: int,
        verbose: bool = True,
    ) -> Tuple[Dict[str, float], str]:
        validation_losses_and_metrics_epoch_all_steps = []
        validation_epoch_metrics_str = ""
        for batch, outputs in zip(val_data_loader, val_outputs_loader):
            validation_losses_and_metrics_current_batch = self.validation_step(
                state,
                batch,
                outputs,
                loss_fun,
                rng,
                val_dataset_size,
                metrics,
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

    def validation_step(
        self,
        state: OutputCalibState,
        batch: Batch,
        outputs: Array,
        loss_fun: Callable,
        rng: PRNGKeyArray,
        n_data: int,
        metrics: Optional[
            Tuple[Callable[[jnp.ndarray, jnp.ndarray, Array], Array], ...]
        ] = None,
    ) -> Dict[str, jnp.ndarray]:
        validation_loss, aux = self.validation_loss_step(
            state, batch, outputs, loss_fun, rng, n_data
        )
        validation_metrics = self.validation_metrics_step(aux, batch, metrics)
        return {"validation_loss": validation_loss, **validation_metrics}

    @abc.abstractmethod
    def validation_loss_step(
        self,
        state: OutputCalibState,
        batch: Batch,
        outputs: Array,
        loss_fun: Callable,
        rng: PRNGKeyArray,
        n_data: int,
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        pass

    def validation_metrics_step(
        self,
        aux: Dict[str, jnp.ndarray],
        batch: Batch,
        metrics: Optional[
            Tuple[Callable[[jnp.ndarray, jnp.ndarray, Array], Array], ...]
        ] = None,
    ) -> Dict[str, jnp.ndarray]:
        if metrics is not None:
            validation_metrics = self.compute_metrics(
                self.predict_fn(aux["outputs"]),
                self.uncertainty_fn(aux["outputs"]),
                batch[1],
                metrics,
            )
            return {f"validation_{m}": v for m, v in validation_metrics.items()}
        else:
            return {}

    def training_epoch_end(
        self, training_losses_and_metrics_current_epoch: List[Dict[str, jnp.ndarray]]
    ) -> Dict[str, float]:
        return self._get_mean_losses_and_metrics(
            training_losses_and_metrics_current_epoch
        )

    def validation_epoch_end(
        self,
        validation_losses_and_metrics_current_epoch: List[Dict[str, jnp.ndarray]],
        state: OutputCalibState,
    ) -> Dict[str, float]:
        validation_losses_and_metrics_current_epoch = self._get_mean_losses_and_metrics(
            validation_losses_and_metrics_current_epoch
        )
        # early stopping
        improved = self.early_stopping_update(
            validation_losses_and_metrics_current_epoch
        )
        if improved and self.save_checkpoint_dir is not None:
            self.save_checkpoint(state, self.save_checkpoint_dir, force_save=True)
        return validation_losses_and_metrics_current_epoch

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
        state: OutputCalibState,
        data_loaders: List[DataLoader],
        outputs_loaders: List[TargetsLoader],
        rng: PRNGKeyArray,
    ) -> Tuple[OutputCalibState, List[DataLoader], List[TargetsLoader], PRNGKeyArray]:
        return state, data_loaders, outputs_loaders, rng

    def on_train_end(self, state: OutputCalibState) -> OutputCalibState:
        self.save_checkpoint(
            state,
            save_checkpoint_dir=self.save_checkpoint_dir,
            keep=self.keep_top_n_checkpoints,
            force_save=True,
        )
        return state

    def on_validation_start(self, state: OutputCalibState) -> OutputCalibState:
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
