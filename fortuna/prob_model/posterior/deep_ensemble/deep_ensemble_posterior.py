from __future__ import annotations

import logging
import os
import pathlib
from typing import (
    List,
    Optional,
    Tuple,
    Type,
)

from flax.core import FrozenDict
from jax import (
    pure_callback,
    random,
)
from jax._src.prng import PRNGKeyArray

from fortuna.data.loader import DataLoader
from fortuna.prob_model.fit_config.base import FitConfig
from fortuna.prob_model.joint.base import Joint
from fortuna.prob_model.joint.state import JointState
from fortuna.prob_model.posterior.base import Posterior
from fortuna.prob_model.posterior.deep_ensemble import DEEP_ENSEMBLE_NAME
from fortuna.prob_model.posterior.deep_ensemble.deep_ensemble_approximator import (
    DeepEnsemblePosteriorApproximator,
)
from fortuna.prob_model.posterior.map.map_posterior import MAPState
from fortuna.prob_model.posterior.map.map_trainer import (
    JittedMAPTrainer,
    MAPTrainer,
    MultiDeviceMAPTrainer,
)
from fortuna.prob_model.posterior.posterior_multi_state_repository import (
    PosteriorMultiStateRepository,
)
from fortuna.prob_model.posterior.run_preliminary_map import run_preliminary_map
from fortuna.typing import (
    Path,
    Status,
)
from fortuna.utils.builtins import get_dynamic_scale_instance_from_model_dtype
from fortuna.utils.device import select_trainer_given_devices
from fortuna.utils.freeze import get_trainable_paths
from fortuna.utils.nested_dicts import (
    nested_get,
    nested_set,
)

logger = logging.getLogger(__name__)


class DeepEnsemblePosterior(Posterior):
    def __init__(
        self,
        joint: Joint,
        posterior_approximator: DeepEnsemblePosteriorApproximator,
    ):
        """
        Deep ensemble approximate posterior class.

        Parameters
        ----------
        joint: Joint
            Joint distribution.
        posterior_approximator: DeepEnsemble
            Deep ensemble posterior approximator.
        """
        super().__init__(joint=joint, posterior_approximator=posterior_approximator)

    def __str__(self):
        return DEEP_ENSEMBLE_NAME

    def fit(
        self,
        train_data_loader: DataLoader,
        val_data_loader: Optional[DataLoader] = None,
        fit_config: FitConfig = FitConfig(),
        map_fit_config: Optional[FitConfig] = None,
        **kwargs,
    ) -> List[Status]:
        super()._checks_on_fit_start(fit_config, map_fit_config)

        status = dict()

        map_state = None
        if map_fit_config is not None and fit_config.optimizer.freeze_fun is None:
            logging.warning(
                "It appears that you are trying to configure `map_fit_config`. "
                "However, a preliminary run with MAP is supported only if "
                "`fit_config.optimizer.freeze_fun` is given. "
                "Since the latter was not given, `map_fit_config` will be ignored."
            )
        elif not super()._is_state_available_somewhere(
            fit_config
        ) and super()._should_run_preliminary_map(fit_config, map_fit_config):
            map_state, status["map"] = run_preliminary_map(
                joint=self.joint,
                train_data_loader=train_data_loader,
                val_data_loader=val_data_loader,
                map_fit_config=map_fit_config,
                rng=self.rng,
                **kwargs,
            )

        trainer_cls = select_trainer_given_devices(
            devices=fit_config.processor.devices,
            base_trainer_cls=MAPTrainer,
            jitted_trainer_cls=JittedMAPTrainer,
            multi_device_trainer_cls=MultiDeviceMAPTrainer,
            disable_jit=fit_config.processor.disable_jit,
        )

        train_data_size = train_data_loader.size
        val_data_size = val_data_loader.size if val_data_loader is not None else None

        def _fit(i):
            if self._is_state_available_somewhere(fit_config):
                _state = self._restore_state_from_somewhere(
                    i=i,
                    fit_config=fit_config,
                    allowed_states=(MAPState,),
                )
            else:
                _state = self._init_map_state(map_state, train_data_loader, fit_config)

            _state = self._freeze_optimizer_in_state(_state, fit_config)

            save_checkpoint_dir_i = (
                pathlib.Path(fit_config.checkpointer.save_checkpoint_dir) / str(i)
                if fit_config.checkpointer.save_checkpoint_dir
                else None
            )
            trainer = trainer_cls(
                predict_fn=self.joint.likelihood.prob_output_layer.predict,
                save_checkpoint_dir=save_checkpoint_dir_i,
                save_every_n_steps=fit_config.checkpointer.save_every_n_steps,
                keep_top_n_checkpoints=fit_config.checkpointer.keep_top_n_checkpoints,
                disable_training_metrics_computation=fit_config.monitor.disable_training_metrics_computation,
                eval_every_n_epochs=fit_config.monitor.eval_every_n_epochs,
                early_stopping_monitor=fit_config.monitor.early_stopping_monitor,
                early_stopping_min_delta=fit_config.monitor.early_stopping_min_delta,
                early_stopping_patience=fit_config.monitor.early_stopping_patience,
            )

            return trainer.train(
                rng=self.rng.get(),
                state=_state,
                loss_fun=self.joint._batched_negative_log_joint_prob,
                training_dataloader=train_data_loader,
                training_dataset_size=train_data_size,
                n_epochs=fit_config.optimizer.n_epochs,
                metrics=fit_config.monitor.metrics,
                validation_dataloader=val_data_loader,
                validation_dataset_size=val_data_size,
                verbose=fit_config.monitor.verbose,
                callbacks=fit_config.callbacks,
                max_grad_norm=fit_config.hyperparameters.max_grad_norm,
                gradient_accumulation_steps=fit_config.hyperparameters.gradient_accumulation_steps,
            )

        if isinstance(self.state, PosteriorMultiStateRepository):
            for i in range(self.posterior_approximator.ensemble_size):
                self.state.state[i].checkpoint_dir = (
                    pathlib.Path(fit_config.checkpointer.save_checkpoint_dir) / str(i)
                    if fit_config.checkpointer.save_checkpoint_dir is not None
                    and fit_config.checkpointer.dump_state
                    else None
                )
        else:
            self.state = PosteriorMultiStateRepository(
                size=self.posterior_approximator.ensemble_size,
                checkpoint_dir=fit_config.checkpointer.save_checkpoint_dir
                if fit_config.checkpointer.dump_state is True
                else None,
            )

        status = []
        for i in range(self.posterior_approximator.ensemble_size):
            logging.info(
                f"Run {i+1} out of {self.posterior_approximator.ensemble_size}."
            )
            state, _status = _fit(i)
            self.state.put(
                state=state, i=i, keep=fit_config.checkpointer.keep_top_n_checkpoints
            )
            status.append(_status)
        logging.info("Fit completed.")
        return status

    def sample(self, rng: Optional[PRNGKeyArray] = None, **kwargs) -> JointState:
        if rng is None:
            rng = self.rng.get()
        state = pure_callback(
            lambda j: self.state.get(i=j),
            self.state.get(i=0),
            random.choice(rng, self.posterior_approximator.ensemble_size),
        )
        return JointState(
            params=state.params,
            mutable=state.mutable,
            calib_params=state.calib_params,
            calib_mutable=state.calib_mutable,
        )

    def load_state(self, checkpoint_dir: Path) -> None:
        try:
            self.restore_checkpoint(pathlib.Path(checkpoint_dir) / "0")
        except ValueError:
            raise ValueError(
                f"No checkpoint was found in `checkpoint_dir={checkpoint_dir}`."
            )
        self.state = PosteriorMultiStateRepository(
            size=self.posterior_approximator.ensemble_size,
            checkpoint_dir=checkpoint_dir,
        )

    def save_state(self, checkpoint_dir: Path, keep_top_n_checkpoints: int = 1) -> None:
        for i in range(self.posterior_approximator.ensemble_size):
            self.state.put(state=self.state.get(i), i=i, keep=keep_top_n_checkpoints)

    def _init_map_state(
        self, state: Optional[MAPState], data_loader: DataLoader, fit_config: FitConfig
    ) -> MAPState:
        if state is None or fit_config.optimizer.freeze_fun is None:
            state = super()._init_joint_state(data_loader)

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
        else:
            random_state = super()._init_joint_state(data_loader)
            trainable_paths = get_trainable_paths(
                state.params, fit_config.optimizer.freeze_fun
            )
            state = state.replace(
                params=FrozenDict(
                    nested_set(
                        d=state.params.unfreeze(),
                        key_paths=trainable_paths,
                        objs=tuple(
                            [
                                nested_get(d=random_state.params, keys=path)
                                for path in trainable_paths
                            ]
                        ),
                    )
                )
            )

        return state

    def _restore_state_from_somewhere(
        self,
        i: int,
        fit_config: FitConfig,
        allowed_states: Optional[Tuple[Type[MAPState], ...]] = None,
    ) -> MAPState:
        if fit_config.checkpointer.restore_checkpoint_path is not None:
            restore_checkpoint_path = pathlib.Path(
                fit_config.checkpointer.restore_checkpoint_path
            ) / str(i)
            state = self.restore_checkpoint(
                restore_checkpoint_path=restore_checkpoint_path,
                optimizer=fit_config.optimizer.method,
            )
        elif fit_config.checkpointer.start_from_current_state is not None:
            state = self.state.get(i=i, optimizer=fit_config.optimizer.method)

        if allowed_states is not None and not isinstance(state, allowed_states):
            raise ValueError(
                f"The type of the restored checkpoint must be within {allowed_states}. "
                f"However, {fit_config.checkpointer.restore_checkpoint_path} pointed to a state "
                f"with type {type(state)}."
            )

        self._check_state(state)
        return state
