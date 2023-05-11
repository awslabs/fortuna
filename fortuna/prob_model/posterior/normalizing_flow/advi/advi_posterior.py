from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict
from jax._src.prng import PRNGKeyArray
from jax.flatten_util import ravel_pytree

from fortuna.data.loader import DataLoader, InputsLoader
from fortuna.distribution.gaussian import DiagGaussian
from fortuna.prob_model.fit_config.base import FitConfig
from fortuna.prob_model.joint.base import Joint
from fortuna.prob_model.joint.state import JointState
from fortuna.prob_model.posterior.base import Posterior
from fortuna.prob_model.posterior.map.map_state import MAPState
from fortuna.prob_model.posterior.normalizing_flow.advi import ADVI_NAME
from fortuna.prob_model.posterior.normalizing_flow.advi.advi_approximator import (
    ADVIPosteriorApproximator,
)
from fortuna.prob_model.posterior.normalizing_flow.advi.advi_architecture import (
    ADVIArchitecture,
)
from fortuna.prob_model.posterior.normalizing_flow.advi.advi_state import ADVIState
from fortuna.prob_model.posterior.normalizing_flow.advi.advi_trainer import (
    ADVITrainer,
    JittedADVITrainer,
    MultiDeviceADVITrainer,
)
from fortuna.prob_model.posterior.posterior_state_repository import (
    PosteriorStateRepository,
)
from fortuna.prob_model.posterior.run_preliminary_map import run_preliminary_map
from fortuna.typing import AnyKey, Array, OptaxOptimizer, Params, Status
from fortuna.utils.device import select_trainer_given_devices
from fortuna.utils.freeze import get_trainable_paths
from fortuna.utils.nested_dicts import nested_get, nested_set, nested_unpair
from fortuna.utils.strings import decode_encoded_tuple_of_lists_of_strings_to_array


class ADVIPosterior(Posterior):
    def __init__(
        self,
        joint: Joint,
        posterior_approximator: ADVIPosteriorApproximator,
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
        map_fit_config: Optional[FitConfig] = None,
        **kwargs,
    ) -> Dict[str, Status]:
        super()._checks_on_fit_start(fit_config, map_fit_config)

        status = dict()

        if super()._is_state_available_somewhere(fit_config):
            state = super()._restore_state_from_somewhere(
                fit_config=fit_config, allowed_states=(MAPState, ADVIState)
            )

        elif super()._should_run_preliminary_map(fit_config, map_fit_config):
            state, status["map"] = run_preliminary_map(
                joint=self.joint,
                train_data_loader=train_data_loader,
                val_data_loader=val_data_loader,
                map_fit_config=map_fit_config,
                rng=self.rng,
                **kwargs,
            )
        else:
            state = None

        state, log_stds = self._init_map_state(
            state=state, data_loader=train_data_loader, fit_config=fit_config
        )

        if fit_config.optimizer.freeze_fun is not None:
            which_params = get_trainable_paths(
                params=state.params, freeze_fun=fit_config.optimizer.freeze_fun
            )
        else:
            which_params = None

        rav, self._unravel, self._indices, rav_log_stds = self._get_unravel(
            params=state.params, log_stds=log_stds, which_params=which_params
        )

        size_rav = len(rav)
        self._base, self._architecture = self._get_base_and_architecture(size_rav)

        trainer_cls = select_trainer_given_devices(
            devices=fit_config.processor.devices,
            BaseTrainer=ADVITrainer,
            JittedTrainer=JittedADVITrainer,
            MultiDeviceTrainer=MultiDeviceADVITrainer,
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
            base=self._base,
            architecture=self._architecture,
            which_params=which_params,
            all_params=state.params if which_params else None,
            indices=self._indices,
            unravel=self._unravel,
        )

        state = self._init_advi_from_map_state(
            rav=rav,
            rav_log_stds=rav_log_stds,
            state=state,
            init_params=self._architecture.init_params,
            optimizer=fit_config.optimizer.method,
        )

        logging.info("Run ADVI.")
        state, status["advi"] = trainer.train(
            rng=self.rng.get(),
            state=state,
            loss_fun=self.joint._batched_negative_log_joint_prob,
            training_dataloader=train_data_loader,
            training_dataset_size=train_data_loader.size,
            n_epochs=fit_config.optimizer.n_epochs,
            metrics=fit_config.monitor.metrics,
            validation_dataloader=val_data_loader,
            validation_dataset_size=val_data_loader.size
            if val_data_loader is not None
            else None,
            verbose=fit_config.monitor.verbose,
            unravel=self._unravel,
            n_samples=self.posterior_approximator.n_loss_samples,
            callbacks=fit_config.callbacks,
        )
        trainer._all_params = None

        self.state = PosteriorStateRepository(
            fit_config.checkpointer.save_checkpoint_dir
            if fit_config.checkpointer.dump_state is True
            else None
        )
        self.state.put(state, keep=fit_config.checkpointer.keep_top_n_checkpoints)
        logging.info("Fit completed.")
        return status

    def _get_base_and_architecture(
        self, size_rav: int
    ) -> Tuple[DiagGaussian, ADVIArchitecture]:
        base = DiagGaussian(
            mean=jnp.zeros(size_rav),
            std=jnp.exp(self.posterior_approximator.log_std_base) * jnp.ones(size_rav),
        )
        architecture = ADVIArchitecture(
            size_rav, std_init_params=self.posterior_approximator.std_init_params
        )
        return base, architecture

    def sample(
        self,
        rng: Optional[PRNGKeyArray] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
        inputs_loader: Optional[InputsLoader] = None,
        inputs: Optional[Array] = None,
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
            Input data loader. If `input_shape` is passed, then `inputs` and `inputs_loader` are ignored.
        inputs: Optional[Array]
            Input variables.
        Returns
        -------
        JointState
            A sample from the posterior distribution.
        """
        if rng is None:
            rng = self.rng.get()
        state = self.state.get()

        if not hasattr(self, "_base") or not hasattr(self, "_unravel"):
            if state._encoded_which_params is None:
                n_params = len(ravel_pytree(state.params)[0]) // 2
                which_params = None
            else:
                which_params = decode_encoded_tuple_of_lists_of_strings_to_array(
                    state._encoded_which_params
                )
                n_params = len(
                    ravel_pytree(
                        nested_unpair(
                            d=state.params.unfreeze(),
                            key_paths=which_params,
                            labels=("mean", "log_std"),
                        )[1]
                    )[0]
                )
            self._base, self._architecture = self._get_base_and_architecture(n_params)
            self._unravel, self._indices = self._get_unravel(
                params=nested_unpair(
                    d=state.params.unfreeze(),
                    key_paths=which_params,
                    labels=("mean", "log_std"),
                )[0]
                if which_params
                else {
                    k: dict(params=v["params"]["mean"]) for k, v in state.params.items()
                },
                which_params=which_params,
            )[1:3]

        if state._encoded_which_params is None:
            means = self._unravel(
                self._architecture.forward(
                    {
                        s: ravel_pytree(
                            {k: v["params"][s] for k, v in state.params.items()}
                        )[0]
                        for s in ["mean", "log_std"]
                    },
                    self._base.sample(rng),
                )[0][0]
            )
        else:
            which_params = decode_encoded_tuple_of_lists_of_strings_to_array(
                state._encoded_which_params
            )
            means, log_stds = nested_unpair(
                d=state.params.unfreeze(),
                key_paths=which_params,
                labels=("mean", "log_std"),
            )
            rav_params = {
                k: ravel_pytree([nested_get(d=d, keys=path) for path in which_params])[
                    0
                ]
                for k, d in zip(["mean", "log_std"], [means, log_stds])
            }
            rav_params = self._architecture.forward(
                params=rav_params, u=self._base.sample(rng)
            )[0][0]

            means = FrozenDict(
                nested_set(
                    d=means,
                    key_paths=which_params,
                    objs=tuple(
                        [
                            _unravel(
                                rav_params[self._indices[i] : self._indices[i + 1]]
                            )
                            for i, _unravel in enumerate(self._unravel)
                        ]
                    ),
                )
            )

        return JointState(
            params=FrozenDict(means),
            mutable=state.mutable,
            calib_params=state.calib_params,
            calib_mutable=state.calib_mutable,
        )

    def _init_map_state(
        self,
        state: Optional[Union[MAPState, ADVIState]],
        data_loader: DataLoader,
        fit_config: FitConfig,
    ) -> Tuple[MAPState, Params]:
        if state is None:
            state = super()._init_joint_state(data_loader)

        log_stds = None

        if isinstance(state, ADVIState):
            if state._encoded_which_params is not None:
                which_params = decode_encoded_tuple_of_lists_of_strings_to_array(
                    state._encoded_which_params
                )
                means, log_stds = nested_unpair(
                    d=state.params.unfreeze(),
                    key_paths=which_params,
                    labels=("mean", "log_std"),
                )
                means, log_stds = FrozenDict(means), FrozenDict(log_stds)
            else:
                means, log_stds = [
                    FrozenDict(
                        {
                            k: dict(params=v["params"][s])
                            for k, v in state.params.items()
                        }
                    )
                    for s in ["mean", "log_std"]
                ]
            state = state.replace(params=means)
            del means

        state = MAPState.init(
            params=state.params,
            mutable=state.mutable,
            optimizer=fit_config.optimizer.method,
            calib_params=state.calib_params,
            calib_mutable=state.calib_mutable,
        )

        return state, log_stds

    def _init_advi_from_map_state(
        self,
        rav: Array,
        rav_log_stds: Optional[Array],
        state: MAPState,
        init_params: Callable,
        optimizer: OptaxOptimizer,
    ) -> ADVIState:
        return ADVIState.init(
            params=FrozenDict(
                init_params(self.rng.get(), mean=rav, log_std=rav_log_stds),
            ),
            mutable=state.mutable,
            optimizer=optimizer,
            calib_params=state.calib_params,
            calib_mutable=state.calib_mutable,
        )

    def _get_unravel(
        self,
        params,
        log_stds: Optional[Params] = None,
        which_params: Optional[Tuple[List[AnyKey], ...]] = None,
    ):
        if which_params is None:
            rav, unravel = ravel_pytree(params)
            rav_log_stds = ravel_pytree(log_stds)[0] if log_stds is not None else None
            indices = None
        else:

            def unravel_fn(_params, _path):
                return ravel_pytree(nested_get(_params, _path))

            rav, unravel, indices, rav_log_stds = [], [], [], []

            for path in which_params:
                _rav, _unravel = unravel_fn(params, path)
                unravel.append(_unravel)
                rav.append(_rav)

                if log_stds is not None:
                    rav_log_stds.append(unravel_fn(log_stds, path)[0])

                indices.append(len(_rav))

            rav = jnp.concatenate(rav)

            if log_stds is not None:
                rav_log_stds = jnp.concatenate(rav_log_stds)
            else:
                rav_log_stds = None

            unravel = tuple(unravel)
            indices = np.concatenate((np.array([0]), np.cumsum(indices)))

        return rav, unravel, indices, rav_log_stds
