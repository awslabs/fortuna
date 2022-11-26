from __future__ import annotations

import logging
from typing import Optional

import jax.numpy as jnp
from jax import random
from jax._src.prng import PRNGKeyArray
from jax.flatten_util import ravel_pytree

from fortuna.data.loader import DataLoader, InputsLoader
from fortuna.prob_model.fit_config import FitConfig
from fortuna.prob_model.joint.base import Joint
from fortuna.prob_model.joint.state import JointState
from fortuna.prob_model.posterior.base import Posterior
from fortuna.prob_model.posterior.map.map_approximator import \
    MAPPosteriorApproximator
from fortuna.prob_model.posterior.map.map_posterior import MAPPosterior
from fortuna.prob_model.posterior.map.map_state import MAPState
from fortuna.prob_model.posterior.posterior_state_repository import \
    PosteriorStateRepository
from fortuna.prob_model.posterior.swag import SWAG_NAME
from fortuna.prob_model.posterior.swag.swag_approximator import \
    SWAGPosteriorApproximator
from fortuna.prob_model.posterior.swag.swag_state import SWAGState
from fortuna.prob_model.posterior.swag.swag_trainer import (
    JittedSWAGTrainer, MultiGPUSWAGTrainer, SWAGTrainer)
from fortuna.typing import Array, Status
from fortuna.utils.gpu import select_trainer_given_devices


class SWAGPosterior(Posterior):
    def __init__(self, joint: Joint, posterior_approximator: SWAGPosteriorApproximator):
        """
        SWAG approximate posterior class.

        Parameters
        ----------
        joint: Joint
            A joint distribution object.
        posterior_approximator: SWAGPosteriorApproximator
            A SWAG posterior approximator.
        """
        super().__init__(joint=joint, posterior_approximator=posterior_approximator)

    def __str__(self):
        return SWAG_NAME

    def fit(
        self,
        train_data_loader: DataLoader,
        val_data_loader: Optional[DataLoader] = None,
        fit_config: FitConfig = FitConfig(),
        map_fit_config: Optional[FitConfig] = None,
        *kwargs,
    ) -> Status:
        if (
            fit_config.checkpointer.dump_state is True
            and not fit_config.checkpointer.save_checkpoint_dir
        ):
            raise ValueError(
                "`save_checkpoint_dir` must be passed when `dump_state` is set to True."
            )
        if self.posterior_approximator.rank < 2:
            raise ValueError("`rank` must be at least 2.")
        if fit_config.optimizer.n_epochs <= self.posterior_approximator.rank:
            raise ValueError(
                """Not enough SWAG epochs to obtain `rank={}`. Please either increase `n_swag_epochs` or 
            decrease `rank`.""".format(
                    self.posterior_approximator.rank
                )
            )
        if (
            fit_config.monitor.early_stopping_patience
            and fit_config.monitor.early_stopping_patience > 0
        ):
            logging.warning(
                f"""It seems you are trying to enable early stopping, since 
            `fit_config.monitor.early_stopping_patience={fit_config.monitor.early_stopping_patience}`. We do not 
            support early stopping in SWAG, since we implement it as a post-processing step of MAP. If your intention
            was rather to enable early stopping in MAP, please configure `map_fit_config` accordingly."""
            )

        init_prob_model_state, n_train_data, n_val_data = self._init(
            train_data_loader, val_data_loader
        )

        status = {}

        if not fit_config.checkpointer.restore_checkpoint_path:
            map_posterior = MAPPosterior(
                self.joint, posterior_approximator=MAPPosteriorApproximator()
            )
            map_posterior.rng = self.rng
            logging.info("Do a preliminary run of MAP.")
            status["map"] = map_posterior.fit(
                train_data_loader=train_data_loader,
                val_data_loader=val_data_loader,
                fit_config=map_fit_config
                if map_fit_config is not None
                else FitConfig(),
            )
            state = SWAGState.convert_from_map_state(
                map_state=map_posterior.state.get(),
                optimizer=fit_config.optimizer.method,
            )
            logging.info("Preliminary run with MAP completed.")
        else:
            state = self.restore_checkpoint(
                restore_checkpoint_path=fit_config.checkpointer.restore_checkpoint_path,
                optimizer=fit_config.optimizer.method,
            )
            if type(state) == MAPState:
                state = SWAGState.convert_from_map_state(
                    map_state=state, optimizer=fit_config.optimizer.method
                )

        trainer_cls = select_trainer_given_devices(
            gpus=fit_config.processor.gpus,
            BaseTrainer=SWAGTrainer,
            JittedTrainer=JittedSWAGTrainer,
            MultiGPUTrainer=MultiGPUSWAGTrainer,
            disable_jit=fit_config.processor.disable_jit,
        )
        trainer = trainer_cls(
            predict_fn=self.joint.likelihood.prob_output_layer.predict,
            save_checkpoint_dir=fit_config.checkpointer.save_checkpoint_dir,
            save_every_n_steps=fit_config.checkpointer.save_every_n_steps,
            keep_top_n_checkpoints=fit_config.checkpointer.keep_top_n_checkpoints,
            disable_training_metrics_computation=fit_config.monitor.disable_training_metrics_computation,
            eval_every_n_epochs=fit_config.monitor.eval_every_n_epochs,
            early_stopping_verbose=False,
        )

        kwargs = dict(rank=self.posterior_approximator.rank)
        logging.info("Run SWAG.")
        state, status["swag"] = trainer.train(
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
            **kwargs,
        )

        self.state = PosteriorStateRepository(
            fit_config.checkpointer.save_checkpoint_dir
            if fit_config.checkpointer.dump_state is True
            else None
        )
        self.state.put(state, keep=fit_config.checkpointer.keep_top_n_checkpoints)
        logging.info("Fit completed.")
        return status

    def sample(
        self,
        rng: Optional[PRNGKeyArray] = None,
        inputs_loader: Optional[InputsLoader] = None,
        inputs: Optional[Array] = None,
        **kwargs,
    ) -> JointState:
        """
        Sample from the posterior distribution.

        Parameters
        ----------
        rng : Optional[PRNGKeyArray]
            A random number generator. If not passed, this will be taken from the attributes of this class.
        inputs_loader: Optional[InputsLoader]
            Input data loader. This or `inputs` is required if the posterior state includes mutable objects.
        inputs: Optional[Array]
            Input variables. This or `inputs_loader` is required if the posterior state includes mutable objects.

        Returns
        -------
        JointState
            A sample from the posterior distribution.
        """
        if rng is None:
            rng = self.rng.get()
        state = self.state.get()
        if state.mutable is not None and inputs_loader is None and inputs is None:
            raise ValueError(
                "The posterior state contains mutable objects. Please pass `inputs_loader` or `inputs`."
            )

        n_params = len(state.mean)
        unravel = ravel_pytree(state.params)[1]
        coeff1 = 1 / jnp.sqrt(2)
        coeff2 = coeff1 / jnp.sqrt(self.posterior_approximator.rank - 1)

        rng, key1, key2 = random.split(rng, 3)
        z1 = random.normal(key1, shape=(n_params,))
        z2 = random.normal(key2, shape=(self.posterior_approximator.rank,))
        state = state.replace(
            params=unravel(
                state.mean
                + coeff1 * state.std * z1
                + coeff2 * jnp.matmul(state.dev, z2)
            )
        )

        if state.mutable:
            if inputs_loader is not None:
                for batch_inputs in inputs_loader:
                    state = state.replace(
                        mutable=self.joint.likelihood.model_manager.apply(
                            state.params,
                            batch_inputs,
                            mutable=state.mutable,
                            train=True,
                            rng=rng,
                        )[1]["mutable"]
                    )
            else:
                state = state.replace(
                    mutable=self.joint.likelihood.model_manager.apply(
                        state.params, inputs, mutable=state.mutable, train=True, rng=rng
                    )[1]["mutable"]
                )

        return JointState(
            params=state.params,
            mutable=state.mutable,
            calib_params=state.calib_params,
            calib_mutable=state.calib_mutable,
        )
