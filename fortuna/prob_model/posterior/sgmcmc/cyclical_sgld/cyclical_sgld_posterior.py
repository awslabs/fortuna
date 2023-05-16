import logging
from typing import Optional
import pathlib

from flax.core import FrozenDict
from fortuna.utils.freeze import get_trainable_paths
from fortuna.utils.nested_dicts import nested_set, nested_get
from fortuna.data.loader import DataLoader
from fortuna.prob_model.fit_config.base import FitConfig
from fortuna.prob_model.joint.base import Joint
from fortuna.prob_model.posterior.map.map_state import MAPState
from fortuna.prob_model.posterior.map.map_trainer import (
    MAPTrainer,
    JittedMAPTrainer,
    MultiDeviceMAPTrainer,
)
from fortuna.prob_model.posterior.run_preliminary_map import (
    run_preliminary_map,
)
from fortuna.prob_model.posterior.posterior_multi_state_repository import (
    PosteriorMultiStateRepository,
)
from fortuna.prob_model.posterior.sgmcmc.sgmcmc_posterior import (
    SGMCMCPosterior,
)
from fortuna.prob_model.posterior.sgmcmc.cyclical_sgld import (
    CYCLICAL_SGLD_NAME,
)
from fortuna.prob_model.posterior.sgmcmc.cyclical_sgld.cyclical_sgld_approximator import (
    CyclicalSGLDPosteriorApproximator,
)
from fortuna.prob_model.posterior.sgmcmc.cyclical_sgld.cyclical_sgld_integrator import (
    cyclical_sgld_integrator,
)
from fortuna.prob_model.posterior.sgmcmc.cyclical_sgld.cyclical_sgld_callback import (
    CyclicalSGLDSamplingCallback,
)
from fortuna.prob_model.posterior.sgmcmc.cyclical_sgld.cyclical_sgld_state import (
    CyclicalSGLDState,
)
from fortuna.typing import Status
from fortuna.utils.device import select_trainer_given_devices

logger = logging.getLogger(__name__)


class CyclicalSGLDPosterior(SGMCMCPosterior):
    def __init__(
        self,
        joint: Joint,
        posterior_approximator: CyclicalSGLDPosteriorApproximator,
    ):
        """
        Cyclical Stochastic Gradient Langevin Dynamics (SGLD) approximate posterior class.

        Parameters
        ----------
        joint: Joint
            A Joint distribution object.
        posterior_approximator: CyclicalSGLDPosteriorApproximator
            A cyclical SGLD posterior approximator.
        """
        super().__init__(joint=joint, posterior_approximator=posterior_approximator)

    def __str__(self):
        return CYCLICAL_SGLD_NAME

    def fit(
        self,
        train_data_loader: DataLoader,
        val_data_loader: Optional[DataLoader] = None,
        fit_config: FitConfig = FitConfig(),
        map_fit_config: Optional[FitConfig] = None,
        **kwargs,
    ) -> Status:
        super()._checks_on_fit_start(fit_config, map_fit_config)

        status = {}

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

        if fit_config.optimizer.method is not None:
            logging.info(f"`FitOptimizer` method in CyclicalSGLD is ignored.")

        fit_config.optimizer.method = cyclical_sgld_integrator(
            rng_key=self.rng.get(),
            init_step_size=self.posterior_approximator.init_step_size,
            cycle_length=self.posterior_approximator.cycle_length,
            exploration_ratio=self.posterior_approximator.exploration_ratio,
            preconditioner=self.posterior_approximator.preconditioner,
        )

        trainer_cls = select_trainer_given_devices(
            devices=fit_config.processor.devices,
            base_trainer_cls=MAPTrainer,
            jitted_trainer_cls=JittedMAPTrainer,
            multi_device_trainer_cls=MultiDeviceMAPTrainer,
            disable_jit=fit_config.processor.disable_jit,
        )

        save_checkpoint_dir = (
            pathlib.Path(fit_config.checkpointer.save_checkpoint_dir) / "c"
            if fit_config.checkpointer.save_checkpoint_dir
            else None
        )
        trainer = trainer_cls(
            predict_fn=self.joint.likelihood.prob_output_layer.predict,
            save_checkpoint_dir=save_checkpoint_dir,
            save_every_n_steps=fit_config.checkpointer.save_every_n_steps,
            keep_top_n_checkpoints=fit_config.checkpointer.keep_top_n_checkpoints,
            disable_training_metrics_computation=fit_config.monitor.disable_training_metrics_computation,
            eval_every_n_epochs=fit_config.monitor.eval_every_n_epochs,
            early_stopping_monitor=fit_config.monitor.early_stopping_monitor,
            early_stopping_min_delta=fit_config.monitor.early_stopping_min_delta,
            early_stopping_patience=fit_config.monitor.early_stopping_patience,
        )

        if super()._is_state_available_somewhere(fit_config):
            state = self._restore_state_from_somewhere(fit_config=fit_config)
        else:
            state = self._init_map_state(map_state, train_data_loader, fit_config)

        state = super()._freeze_optimizer_in_state(state, fit_config)

        self.state = PosteriorMultiStateRepository(
            size=self.posterior_approximator.n_samples,
            checkpoint_dir=fit_config.checkpointer.save_checkpoint_dir
            if fit_config.checkpointer.dump_state is True
            else None,
        )

        cyclical_sampling_callback = CyclicalSGLDSamplingCallback(
            n_epochs=fit_config.optimizer.n_epochs,
            n_training_steps=len(train_data_loader),
            n_samples=self.posterior_approximator.n_samples,
            n_thinning=self.posterior_approximator.n_thinning,
            cycle_length=self.posterior_approximator.cycle_length,
            exploration_ratio=self.posterior_approximator.exploration_ratio,
            trainer=trainer,
            state_repository=self.state,
            keep_top_n_checkpoints=fit_config.checkpointer.keep_top_n_checkpoints,
            save_checkpoint_dir=fit_config.checkpointer.save_checkpoint_dir,
        )

        state = CyclicalSGLDState.convert_from_map_state(
            map_state=state,
            optimizer=fit_config.optimizer.method,
        )

        state = super()._freeze_optimizer_in_state(state, fit_config)

        logging.info(f"Run CyclicalSGLD.")
        state, status = trainer.train(
            rng=self.rng.get(),
            state=state,
            loss_fun=self.joint._batched_log_joint_prob,
            training_dataloader=train_data_loader,
            training_dataset_size=train_data_loader.size,
            n_epochs=fit_config.optimizer.n_epochs,
            metrics=fit_config.monitor.metrics,
            validation_dataloader=val_data_loader,
            validation_dataset_size=val_data_loader.size
            if val_data_loader is not None
            else None,
            verbose=fit_config.monitor.verbose,
            callbacks=[cyclical_sampling_callback],
        )
        logging.info("Fit completed.")

        return status

    def _init_map_state(
        self,
        state: Optional[MAPState],
        data_loader: DataLoader,
        fit_config: FitConfig,
    ) -> MAPState:
        if state is None or fit_config.optimizer.freeze_fun is None:
            state = super()._init_joint_state(data_loader)

            return MAPState.init(
                params=state.params,
                mutable=state.mutable,
                optimizer=fit_config.optimizer.method,
                calib_params=state.calib_params,
                calib_mutable=state.calib_mutable,
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
