import logging
from pathlib import Path
from typing import Optional

from jax import eval_shape
from jax._src.prng import PRNGKeyArray

from fortuna.data.loader import DataLoader
from fortuna.partitioner.partition_manager.base import PartitionManager
from fortuna.prob_model.fit_config.base import FitConfig
from fortuna.prob_model.joint.base import Joint
from fortuna.prob_model.joint.state import JointState
from fortuna.prob_model.posterior.base import Posterior
from fortuna.prob_model.posterior.map import MAP_NAME
from fortuna.prob_model.posterior.map.map_approximator import MAPPosteriorApproximator
from fortuna.prob_model.posterior.map.map_state import MAPState
from fortuna.prob_model.posterior.map.map_trainer import (
    MAPTrainer,
    ShardedMAPTrainer,
)
from fortuna.prob_model.posterior.posterior_state_repository import (
    PosteriorStateRepository,
)
from fortuna.typing import Status, Shape
from fortuna.utils.builtins import get_dynamic_scale_instance_from_model_dtype
from fortuna.utils.checkpoint import get_checkpoint_manager

logger = logging.getLogger(__name__)


class MAPPosterior(Posterior):
    def __init__(
        self,
        joint: Joint,
        posterior_approximator: MAPPosteriorApproximator,
        partition_manager: PartitionManager,
    ):
        """
        Maximum-a-Posteriori (MAP) approximate posterior class.

        Parameters
        ----------
        joint: Joint
            A Joint distribution object.
        posterior_approximator: MAPPosteriorApproximator
            A MAP posterior approximator.
        partition_manager: PartitionManager
            An object to manage partitions.
        """
        super().__init__(
            joint=joint,
            posterior_approximator=posterior_approximator,
            partition_manager=partition_manager,
        )

    def __str__(self):
        return MAP_NAME

    def fit(
        self,
        train_data_loader: DataLoader,
        val_data_loader: Optional[DataLoader] = None,
        fit_config: FitConfig = FitConfig(),
        map_fit_config=None,
        **kwargs,
    ) -> Status:
        super()._checks_on_fit_start(fit_config, map_fit_config)

        trainer_cls = (
            ShardedMAPTrainer if not fit_config.processor.disable_jit else MAPTrainer
        )

        trainer = trainer_cls(
            predict_fn=self.joint.likelihood.prob_output_layer.predict,
            partition_manager=self.partition_manager,
            checkpoint_manager=get_checkpoint_manager(
                fit_config.checkpointer.save_checkpoint_dir,
                keep_top_n_checkpoints=fit_config.checkpointer.keep_top_n_checkpoints,
            ),
            save_checkpoint_dir=fit_config.checkpointer.save_checkpoint_dir,
            save_every_n_steps=fit_config.checkpointer.save_every_n_steps,
            keep_top_n_checkpoints=fit_config.checkpointer.keep_top_n_checkpoints,
            disable_training_metrics_computation=fit_config.monitor.disable_training_metrics_computation,
            eval_every_n_epochs=fit_config.monitor.eval_every_n_epochs,
            early_stopping_monitor=fit_config.monitor.early_stopping_monitor,
            early_stopping_min_delta=fit_config.monitor.early_stopping_min_delta,
            early_stopping_patience=fit_config.monitor.early_stopping_patience,
            freeze_fun=fit_config.optimizer.freeze_fun,
        )

        checkpoint_restorer = (
            get_checkpoint_manager(
                str(
                    Path(fit_config.checkpointer.restore_checkpoint_dir)
                    / fit_config.checkpointer.checkpoint_type
                ),
                keep_top_n_checkpoints=fit_config.checkpointer.keep_top_n_checkpoints,
            )
            if fit_config.checkpointer.restore_checkpoint_dir is not None
            else None
        )

        if self._is_state_available_somewhere(fit_config):
            state = self._restore_state_from_somewhere(
                fit_config=fit_config,
                allowed_states=(MAPState,),
                partition_manager=self.partition_manager,
                checkpoint_manager=checkpoint_restorer,
            )
            state = self._freeze_optimizer_in_state(state, fit_config)
            self.partition_manager.shapes_dtypes = eval_shape(lambda: state)
        else:
            input_shape = train_data_loader.input_shape

            def init_state_fn(rng):
                _state = self._init_state(
                    input_shape=input_shape, fit_config=fit_config, rng=rng
                )
                return self._freeze_optimizer_in_state(_state, fit_config)

            state = self.partition_manager.init_sharded_state(
                init_state_fn, self.rng.get()
            )
        self._check_state(state)

        logging.info("Run MAP.")
        state, status = trainer.train(
            rng=self.rng.get(),
            state=state,
            loss_fun=self.joint._batched_negative_log_joint_prob,
            training_data_loader=train_data_loader,
            training_dataset_size=train_data_loader.size,
            n_epochs=fit_config.optimizer.n_epochs,
            metrics=fit_config.monitor.metrics,
            validation_data_loader=val_data_loader,
            validation_dataset_size=val_data_loader.size
            if val_data_loader is not None
            else None,
            verbose=fit_config.monitor.verbose,
            callbacks=fit_config.callbacks,
            max_grad_norm=fit_config.hyperparameters.max_grad_norm,
            gradient_accumulation_steps=fit_config.hyperparameters.gradient_accumulation_steps,
        )
        self.state = PosteriorStateRepository(
            partition_manager=self.partition_manager,
            checkpoint_manager=get_checkpoint_manager(
                checkpoint_dir=str(
                    Path(fit_config.checkpointer.save_checkpoint_dir)
                    / fit_config.checkpointer.checkpoint_type
                ),
                keep_top_n_checkpoints=fit_config.checkpointer.keep_top_n_checkpoints,
            )
            if fit_config.checkpointer.save_checkpoint_dir is not None
            and fit_config.checkpointer.dump_state
            else None,
        )
        if self.state.checkpoint_manager is None:
            self.state.put(state, keep=fit_config.checkpointer.keep_top_n_checkpoints)
        logging.info("Fit completed.")
        return status

    def sample(self, rng: Optional[PRNGKeyArray] = None, **kwargs) -> JointState:
        state = self.state.get()
        return JointState(
            params=state.params,
            mutable=state.mutable,
            calib_params=state.calib_params,
            calib_mutable=state.calib_mutable,
        )

    def _init_state(
        self,
        input_shape: Shape,
        fit_config: FitConfig,
        rng: Optional[PRNGKeyArray] = None,
    ) -> MAPState:
        state = super()._init_joint_state(input_shape=input_shape, rng=rng)
        return MAPState.init(
            params=state.params,
            mutable=state.mutable,
            optimizer=fit_config.optimizer.method,
            calib_params=state.calib_params,
            calib_mutable=state.calib_mutable,
            dynamic_scale=get_dynamic_scale_instance_from_model_dtype(
                getattr(self.joint.likelihood.model_manager.model, "dtype")
                if hasattr(self.joint.likelihood.model_manager.model, "dtype")
                else None
            ),
        )
