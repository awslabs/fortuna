from __future__ import annotations

import logging
from typing import Optional

import jax.numpy as jnp
from jax import random
from jax._src.prng import PRNGKeyArray
from jax.flatten_util import ravel_pytree

from fortuna.data.loader import DataLoader, InputsLoader
from fortuna.prob_model.fit_config.base import FitConfig
from fortuna.prob_model.joint.base import Joint
from fortuna.prob_model.joint.state import JointState
from fortuna.prob_model.posterior.base import Posterior
from fortuna.prob_model.posterior.map.map_state import MAPState
from fortuna.prob_model.posterior.posterior_state_repository import \
    PosteriorStateRepository
from fortuna.prob_model.posterior.swag import SWAG_NAME
from fortuna.prob_model.posterior.swag.swag_approximator import \
    SWAGPosteriorApproximator
from fortuna.prob_model.posterior.swag.swag_state import SWAGState
from fortuna.prob_model.posterior.swag.swag_trainer import (
    JittedSWAGTrainer, MultiDeviceSWAGTrainer, SWAGTrainer)
from fortuna.typing import Array, Status
from fortuna.utils.device import select_trainer_given_devices
from fortuna.prob_model.posterior.run_preliminary_map import run_preliminary_map
from fortuna.utils.freeze import get_trainable_paths
from fortuna.utils.strings import decode_encoded_tuple_of_lists_of_strings_to_array
from fortuna.utils.nested_dicts import nested_get, nested_set
from flax.core import FrozenDict


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
        **kwargs,
    ) -> Status:
        super()._checks_on_fit_start(fit_config, map_fit_config)

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

        status = dict()

        if super()._is_state_available_somewhere(fit_config):
            state = super()._restore_state_from_somewhere(
                fit_config=fit_config,
                allowed_states=(MAPState, SWAGState)
            )

        elif super()._should_run_preliminary_map(fit_config, map_fit_config):
            state, status["map"] = run_preliminary_map(
                joint=self.joint,
                train_data_loader=train_data_loader,
                val_data_loader=val_data_loader,
                map_fit_config=map_fit_config,
                rng=self.rng,
                **kwargs
            )
        else:
            raise ValueError("The SWAG approximation must start from a preliminary run of MAP or an existing "
                             "checkpoint or state. Please configure `map_fit_config`, or "
                             "`fit_config.checkpointer.restore_checkpoint_path`, "
                             "or `fit_config.checkpointer.start_from_current_state`.")

        state = SWAGState.convert_from_map_state(
            map_state=state,
            optimizer=fit_config.optimizer.method,
        )

        state = super()._freeze_optimizer_in_state(state, fit_config)

        if fit_config.optimizer.freeze_fun is not None:
            which_params = get_trainable_paths(state.params, fit_config.optimizer.freeze_fun)
        else:
            which_params = None

        trainer_cls = select_trainer_given_devices(
            devices=fit_config.processor.devices,
            BaseTrainer=SWAGTrainer,
            JittedTrainer=JittedSWAGTrainer,
            MultiDeviceTrainer=MultiDeviceSWAGTrainer,
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
            which_params=which_params
        )

        kwargs = dict(rank=self.posterior_approximator.rank)
        logging.info("Run SWAG.")
        state, status["swag"] = trainer.train(
            rng=self.rng.get(),
            state=state,
            loss_fun=self.joint._batched_negative_log_joint_prob,
            training_dataloader=train_data_loader,
            training_dataset_size=train_data_loader.size,
            n_epochs=fit_config.optimizer.n_epochs,
            metrics=fit_config.monitor.metrics,
            validation_dataloader=val_data_loader,
            validation_dataset_size=val_data_loader.size if val_data_loader is not None else None,
            verbose=fit_config.monitor.verbose,
            callbacks=fit_config.callbacks,
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
        rank = state.dev.shape[1]
        which_params = decode_encoded_tuple_of_lists_of_strings_to_array(state._encoded_which_params)

        unravel = ravel_pytree(
            state.params if which_params is None else [nested_get(state.params, path) for path in which_params]
        )[1]

        coeff1 = 1 / jnp.sqrt(2)
        coeff2 = coeff1 / jnp.sqrt(rank)

        rng, key1, key2 = random.split(rng, 3)
        z1 = random.normal(key1, shape=(n_params,))
        z2 = random.normal(key2, shape=(rank,))
        if which_params is None:
            state = state.replace(
                params=self._get_sample(
                            mean=state.mean,
                            std=state.std,
                            dev=state.dev,
                            z1=z1,
                            z2=z2,
                            coeff1=coeff1,
                            coeff2=coeff2,
                            unravel=unravel
                    )
            )
        else:
            state = state.replace(
                params=FrozenDict(
                    nested_set(
                        d=state.params.unfreeze(),
                        key_paths=which_params,
                        objs=tuple(
                            self._get_sample(
                                mean=state.mean,
                                std=state.std,
                                dev=state.dev,
                                z1=z1,
                                z2=z2,
                                coeff1=coeff1,
                                coeff2=coeff2,
                                unravel=unravel
                            )
                        )
                    )
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

    def _get_sample(self, mean, std, dev, z1, z2, coeff1, coeff2, unravel):
        return unravel(
            mean
            + coeff1 * std * z1
            + coeff2 * jnp.matmul(dev, z2)
        )

    def _get_mean_std_dev(self, state: SWAGState) -> SWAGState:
        var = state._mean_squared_rav_params - state._mean_rav_params**2
        var = jnp.maximum(var, 0.0)
        return state.update(
            dict(
                mean=state._mean_rav_params if not self.multi_device else state._mean_rav_params[None],
                std=jnp.sqrt(var) if not self.multi_device else jnp.sqrt(var)[None],
                dev=state._deviation_rav_params if not self.multi_device else state._deviation_rav_params[None],
            )
        )

