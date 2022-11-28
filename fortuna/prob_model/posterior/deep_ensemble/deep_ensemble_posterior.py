from __future__ import annotations

import logging
import os
import pathlib
from typing import List, Optional

import numpy as np
from jax._src.prng import PRNGKeyArray

from fortuna.data.loader import DataLoader
from fortuna.prob_model.fit_config import FitConfig
from fortuna.prob_model.joint.base import Joint
from fortuna.prob_model.joint.state import JointState
from fortuna.prob_model.posterior.base import Posterior
from fortuna.prob_model.posterior.deep_ensemble import DEEP_ENSEMBLE_NAME
from fortuna.prob_model.posterior.deep_ensemble.deep_ensemble_approximator import \
    DeepEnsemblePosteriorApproximator
from fortuna.prob_model.posterior.deep_ensemble.deep_ensemble_repositories import \
    DeepEnsemblePosteriorStateRepository
from fortuna.prob_model.posterior.map.map_posterior import MAPState
from fortuna.prob_model.posterior.map.map_trainer import (JittedMAPTrainer,
                                                          MAPTrainer,
                                                          MultiDeviceMAPTrainer)
from fortuna.typing import Path, Status
from fortuna.utils.device import select_trainer_given_devices

logger = logging.getLogger(__name__)


class DeepEnsemblePosterior(Posterior):
    def __init__(
        self, joint: Joint, posterior_approximator: DeepEnsemblePosteriorApproximator,
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
        **kwargs,
    ) -> List[Status]:
        if (
            fit_config.checkpointer.dump_state is True
            and not fit_config.checkpointer.save_checkpoint_dir
        ):
            raise ValueError(
                "`save_checkpoint_dir` must be passed when `dump_state` is set to True."
            )

        trainer_cls = select_trainer_given_devices(
            devices=fit_config.processor.devices,
            BaseTrainer=MAPTrainer,
            JittedTrainer=JittedMAPTrainer,
            MultiDeviceTrainer=MultiDeviceMAPTrainer,
            disable_jit=fit_config.processor.disable_jit,
        )

        def _fit(i):
            init_prob_model_state, n_train_data, n_val_data = self._init(
                train_data_loader, val_data_loader
            )

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

            if fit_config.checkpointer.restore_checkpoint_path:
                state = self.restore_checkpoint(
                    restore_checkpoint_path=str(
                        fit_config.checkpointer.restore_checkpoint_path
                    )
                    + "/"
                    + str(i),
                    optimizer=fit_config.optimizer.method,
                )
            else:
                state = MAPState.init(
                    params=init_prob_model_state.params,
                    mutable=init_prob_model_state.mutable,
                    optimizer=fit_config.optimizer.method,
                    calib_params=init_prob_model_state.calib_params,
                    calib_mutable=init_prob_model_state.calib_mutable,
                )

            return trainer.train(
                rng=self.rng.get(),
                state=state,
                fun=self.joint._batched_log_joint_prob,
                training_dataloader=train_data_loader,
                training_dataset_size=n_train_data,
                n_epochs=fit_config.optimizer.n_epochs,
                metrics=fit_config.monitor.metrics,
                validation_dataloader=val_data_loader,
                validation_dataset_size=n_val_data,
                verbose=fit_config.monitor.verbose,
            )

        self.state = DeepEnsemblePosteriorStateRepository(
            ensemble_size=self.posterior_approximator.ensemble_size,
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
        # FIXME: Bug issue in JAX: https://github.com/google/jax/issues/13098
        # if rng is None:
        #     rng = self.rng.get()
        # state = pure_callback(
        #     lambda j: self.state.get(i=j),
        #     self.state.get(i=0),
        #     random.choice(rng, self.posterior_approximator.ensemble_size),
        # )
        state = self.state.get(
            i=np.random.choice(self.posterior_approximator.ensemble_size)
        )
        return JointState(
            params=state.params,
            mutable=state.mutable,
            calib_params=state.calib_params,
            calib_mutable=state.calib_mutable,
        )

    def load_state(self, checkpoint_dir: Path) -> None:
        try:
            self.restore_checkpoint(os.path.join(checkpoint_dir, "0"))
        except ValueError:
            raise ValueError(
                f"No checkpoint was found in `checkpoint_dir={checkpoint_dir}`."
            )
        self.state = DeepEnsemblePosteriorStateRepository(
            ensemble_size=self.posterior_approximator.ensemble_size,
            checkpoint_dir=checkpoint_dir,
        )

    def save_state(self, checkpoint_dir: Path, keep_top_n_checkpoints: int = 1) -> None:
        for i in range(self.posterior_approximator.ensemble_size):
            self.state.put(state=self.state.get(i), i=i, keep=keep_top_n_checkpoints)
