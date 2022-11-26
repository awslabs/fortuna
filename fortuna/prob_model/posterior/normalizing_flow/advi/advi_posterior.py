from __future__ import annotations

import logging
from typing import Optional, Tuple

import jax.numpy as jnp
from flax.core import FrozenDict
from fortuna.data.loader import DataLoader, InputsLoader
from fortuna.distribution.gaussian import DiagGaussian
from fortuna.prob_model.fit_config import FitConfig
from fortuna.prob_model.joint.base import Joint
from fortuna.prob_model.joint.state import JointState
from fortuna.prob_model.posterior.base import Posterior
from fortuna.prob_model.posterior.normalizing_flow.advi import ADVI_NAME
from fortuna.prob_model.posterior.normalizing_flow.advi.advi_approximator import \
    ADVIPosteriorApproximator
from fortuna.prob_model.posterior.normalizing_flow.advi.advi_architecture import \
    ADVIArchitecture
from fortuna.prob_model.posterior.normalizing_flow.advi.advi_state import \
    ADVIState
from fortuna.prob_model.posterior.normalizing_flow.advi.advi_trainer import \
    ADVITrainer
from fortuna.prob_model.posterior.posterior_state_repository import \
    PosteriorStateRepository
from fortuna.training.trainer import JittedMixin, MultiGPUMixin
from fortuna.typing import Status
from fortuna.utils.gpu import select_trainer_given_devices
from jax._src.prng import PRNGKeyArray
from jax.flatten_util import ravel_pytree


class JittedADVITrainer(JittedMixin, ADVITrainer):
    pass


class MultiGPUADVITrainer(MultiGPUMixin, ADVITrainer):
    pass


class ADVIPosterior(Posterior):
    def __init__(
        self, joint: Joint, posterior_approximator: ADVIPosteriorApproximator,
    ):
        """
        Automatic Differentiation Variational Inference (ADVI) approximate posterior class.

        Parameters
        ----------
        joint: Joint
            A joint distribution object.
        posterior_approximator: ADVI
            An ADVI posterior approximator.
        """
        super().__init__(joint=joint, posterior_approximator=posterior_approximator)

    def __str__(self):
        return ADVI_NAME

    def fit(
        self,
        train_data_loader: DataLoader,
        val_data_loader: Optional[DataLoader] = None,
        fit_config: FitConfig = FitConfig(),
        **kwargs,
    ) -> Status:
        if (
            fit_config.checkpointer.save_state is True
            and not fit_config.checkpointer.save_checkpoint_dir
        ):
            raise ValueError(
                "`save_checkpoint_dir` must be passed when `dump_state` is set to True."
            )

        init_prob_model_state, n_train_data, n_val_data = self._init(
            train_data_loader, val_data_loader
        )

        rav, self.unravel = ravel_pytree(init_prob_model_state.params)
        size_rav = len(rav)
        self.base = DiagGaussian(
            mean=jnp.zeros(size_rav),
            std=self.posterior_approximator.std_base * jnp.ones(size_rav),
        )
        self.architecture = ADVIArchitecture(
            size_rav, std_init_params=self.posterior_approximator.std_init_params
        )

        trainer_cls = select_trainer_given_devices(
            gpus=fit_config.processor.gpus,
            BaseTrainer=ADVITrainer,
            JittedTrainer=JittedADVITrainer,
            MultiGPUTrainer=MultiGPUADVITrainer,
            disable_jit=fit_config.processor.disable_jit,
        )

        trainer = trainer_cls(
            predict_fn=self.joint.likelihood.prob_output_layer.predict,
            save_checkpoint_dir=fit_config.checkpointer.save_checkpoint_dir,
            save_every_n_steps=fit_config.checkpointer.save_every_n_steps,
            keep_top_n_checkpoints=fit_config.checkpointer.keep_top_n_checkpoints,
            disable_training_metrics_computation=fit_config.monitor.disable_training_metrics_computation,
            eval_every_n_epochs=fit_config.monitor.eval_every_n_epochs,
            early_stopping_monitor=fit_config.monitor.early_stopping_monitor,
            early_stopping_min_delta=fit_config.monitor.early_stopping_min_delta,
            early_stopping_patience=fit_config.monitor.early_stopping_patience,
            base=self.base,
            architecture=self.architecture,
        )

        state = None
        if fit_config.checkpointer.restore_checkpoint_path:
            state = self.restore_checkpoint(
                restore_checkpoint_path=fit_config.checkpointer.restore_checkpoint_path,
                optimizer=fit_config.optimizer.method,
            )

        if type(state) != ADVIState:
            state = ADVIState.init(
                FrozenDict(
                    zip(
                        ("mean", "logvar"),
                        trainer.init_params(
                            self.rng.get(),
                            mean=ravel_pytree(
                                getattr(state, "params", init_prob_model_state.params)
                            )[0],
                        ),
                    )
                ),
                getattr(state, "mutable", init_prob_model_state.mutable),
                fit_config.optimizer.method,
                getattr(state, "calib_params", init_prob_model_state.calib_params),
                getattr(state, "calib_mutable", init_prob_model_state.calib_mutable),
            )
        logging.info("Run ADVI.")
        state, status = trainer.train(
            rng=self.rng.get(),
            state=state,
            fun=self.joint.batched_log_prob,
            training_dataloader=train_data_loader,
            training_dataset_size=n_train_data,
            n_epochs=fit_config.optimizer.n_epochs,
            metrics=fit_config.monitor.metrics,
            validation_dataloader=val_data_loader,
            validation_dataset_size=n_val_data,
            verbose=fit_config.monitor.verbose,
            unravel=self.unravel,
            n_samples=self.posterior_approximator.n_loss_samples,
        )
        self.state = PosteriorStateRepository(
            fit_config.checkpointer.save_checkpoint_dir
            if fit_config.checkpointer.save_state is True
            else None
        )
        self.state.put(state, keep=fit_config.checkpointer.keep_top_n_checkpoints)
        logging.info("Fit completed.")
        return status

    def sample(
        self,
        rng: Optional[PRNGKeyArray] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
        inputs_loader: Optional[InputsLoader] = None,
        **kwargs,
    ) -> JointState:
        """
        Sample from the posterior distribution. Either `input_shape` or `_inputs_loader` must be passed.

        Parameters
        ----------
        rng : Optional[PRNGKeyArray]
            A random number generator. If not passed, this will be taken from the attributes of this class.
        input_shape: Optional[Tuple[int, ...]]
            Shape of a single input.
        inputs_loader: Optional[InputsLoader]
            Input data loader. If `input_shape` is passed, then `_inputs_loader` is ignored.

        Returns
        -------
        JointState
            A sample from the posterior distribution.
        """
        if rng is None:
            rng = self.rng.get()
        state = self.state.get()
        state = state.replace(params=tuple(state.params.values()))
        n_params = len(state.params[0])
        if not hasattr(self, "base"):
            self.base = DiagGaussian(
                mean=jnp.zeros(n_params),
                std=self.posterior_approximator.std_base * jnp.ones(n_params),
            )
        if not hasattr(self, "architecture"):
            self.architecture = ADVIArchitecture(
                n_params, std_init_params=self.posterior_approximator.std_init_params
            )

        if not hasattr(self, "unravel"):
            if not input_shape:
                if not inputs_loader:
                    raise ValueError(
                        "Either `input_shape` or `_inputs_loader` must be passed."
                    )
                for x in inputs_loader:
                    input_shape = x.shape[1:]
                    break
            model_manager_state = self.joint.likelihood.model_manager.init(input_shape)
            self.unravel = ravel_pytree(model_manager_state.params)[1]
        sample_params = self.unravel(
            self.architecture.forward(state.params, self.base.sample(rng))[0][0]
        )

        return JointState(
            params=FrozenDict(sample_params),
            mutable=state.mutable,
            calib_params=state.calib_params,
            calib_mutable=state.calib_mutable,
        )
