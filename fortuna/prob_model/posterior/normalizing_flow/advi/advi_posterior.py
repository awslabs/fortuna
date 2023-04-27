from __future__ import annotations

import logging
from typing import Optional, Tuple, Dict

import jax.numpy as jnp
from flax.core import FrozenDict
from jax._src.prng import PRNGKeyArray
from jax.flatten_util import ravel_pytree

from fortuna.data.loader import DataLoader, InputsLoader
from fortuna.distribution.gaussian import DiagGaussian
from fortuna.prob_model.fit_config.base import FitConfig
from fortuna.prob_model.joint.base import Joint
from fortuna.prob_model.joint.state import JointState
from fortuna.prob_model.posterior.gaussian_posterior import GaussianPosterior
from fortuna.prob_model.posterior.normalizing_flow.advi import ADVI_NAME
from fortuna.prob_model.posterior.normalizing_flow.advi.advi_approximator import \
    ADVIPosteriorApproximator
from fortuna.prob_model.posterior.normalizing_flow.advi.advi_architecture import \
    ADVIArchitecture
from fortuna.prob_model.posterior.normalizing_flow.advi.advi_state import \
    ADVIState
from fortuna.prob_model.posterior.normalizing_flow.advi.advi_trainer import \
    ADVITrainer, JittedADVITrainer, MultiDeviceADVITrainer
from fortuna.prob_model.posterior.posterior_state_repository import \
    PosteriorStateRepository
from fortuna.typing import Status, Array
from fortuna.utils.device import select_trainer_given_devices
from fortuna.prob_model.posterior.map.map_state import MAPState
from fortuna.utils.nested_dicts import nested_get, nested_unpair, nested_set
from fortuna.prob_model.posterior.map.map_posterior import MAPPosterior
from fortuna.prob_model.posterior.map.map_approximator import MAPPosteriorApproximator
from fortuna.prob_model.posterior.state import PosteriorState
import numpy as np


class ADVIPosterior(GaussianPosterior):
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
        map_fit_config: Optional[FitConfig] = None,
        fit_config: FitConfig = FitConfig(),
        **kwargs,
    ) -> Dict[str, Status]:
        if (
            fit_config.checkpointer.dump_state is True
            and not fit_config.checkpointer.save_checkpoint_dir
        ):
            raise ValueError(
                "`save_checkpoint_dir` must be passed when `dump_state` is set to True."
            )

        log_stds = None

        state, status, n_train_data, n_val_data = self._init_state(
            train_data_loader=train_data_loader,
            val_data_loader=val_data_loader,
            fit_config=fit_config,
            map_fit_config=map_fit_config,
            **kwargs
        )

        rav, self._unravel, self._indices, rav_log_stds = self._get_unravel(state.params, log_stds)

        size_rav = len(rav)
        self.base = DiagGaussian(
            mean=jnp.zeros(size_rav),
            std=jnp.exp(self.posterior_approximator.log_std_base) * jnp.ones(size_rav),
        )
        self.architecture = ADVIArchitecture(
            size_rav, std_init_params=self.posterior_approximator.std_init_params
        )

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
            base=self.base,
            architecture=self.architecture,
            which_params=self.posterior_approximator.which_params,
            all_params=state.params if self.posterior_approximator.which_params else None,
            indices=self._indices,
            unravel=self._unravel
        )

        state = ADVIState.init(
            params=FrozenDict(
                    trainer.init_params(
                        self.rng.get(),
                        mean=rav,
                        log_std=None if log_stds is None else (rav, rav_log_stds)
                    ),
                ),
            mutable=getattr(state, "mutable", state.mutable),
            optimizer=fit_config.optimizer.method,
            calib_params=getattr(state, "calib_params", state.calib_params),
            calib_mutable=getattr(state, "calib_mutable", state.calib_mutable),
        )

        logging.info("Run ADVI.")
        state, status["advi"] = trainer.train(
            rng=self.rng.get(),
            state=state,
            loss_fun=self.joint._batched_negative_log_joint_prob,
            training_dataloader=train_data_loader,
            training_dataset_size=n_train_data,
            n_epochs=fit_config.optimizer.n_epochs,
            metrics=fit_config.monitor.metrics,
            validation_dataloader=val_data_loader,
            validation_dataset_size=n_val_data,
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

        if not hasattr(self, "base") or not hasattr(self, "_unravel"):
            if self.posterior_approximator.which_params is None:
                n_params = len(ravel_pytree(state.params)[0]) // 2
            else:
                n_params = len(ravel_pytree(
                    nested_unpair(
                        d=state.params.unfreeze(),
                        key_paths=self.posterior_approximator.which_params,
                        labels=("mean", "log_std")
                    )[0]
                )[0])
            self.base = DiagGaussian(
                mean=jnp.zeros(n_params),
                std=jnp.exp(self.posterior_approximator.log_std_base) * jnp.ones(n_params),
            )
            self.architecture = ADVIArchitecture(
                n_params, std_init_params=self.posterior_approximator.std_init_params
            )
            self._unravel, self._indices = self._get_unravel(state.params)[1:2]

        if self.posterior_approximator.which_params is None:
            means = self._unravel(
                self.architecture.forward(
                    {s: ravel_pytree({k: v["params"][s] for k, v in state.params.items()})[0] for s in ["mean", "log_std"]},
                    self.base.sample(rng)
                )[0][0]
            )
        else:
            means, log_stds = nested_unpair(
                        d=state.params.unfreeze(),
                        key_paths=self.posterior_approximator.which_params,
                        labels=("mean", "log_std")
                    )
            rav_params = {
                k: ravel_pytree(
                    [nested_get(
                        d=d,
                        keys=path
                    ) for path in self.posterior_approximator.which_params]
                )[0] for k, d in zip(
                    ["mean", "log_std"],
                    [means, log_stds]
                )
            }
            rav_params = self.architecture.forward(
                params=rav_params,
                u=self.base.sample(rng)
            )[0][0]

            means = FrozenDict(
                nested_set(
                    d=means,
                    key_paths=self.posterior_approximator.which_params,
                    objs=tuple([_unravel(rav_params[self._indices[i]:self._indices[i + 1]]) for i, _unravel in enumerate(self._unravel)]),
                )
            )

        return JointState(
            params=FrozenDict(means),
            mutable=state.mutable,
            calib_params=state.calib_params,
            calib_mutable=state.calib_mutable,
        )

    def _init_state(
            self,
            train_data_loader: DataLoader,
            val_data_loader: DataLoader,
            fit_config: FitConfig,
            map_fit_config: Optional[FitConfig] = None,
            **kwargs
    ) -> Tuple[PosteriorState, Dict[str, Status], int, Optional[int]]:
        allowed_states = (MAPState, ADVIState)
        state = None
        status = dict()

        if fit_config.checkpointer.restore_checkpoint_path is not None:
            state = self.restore_checkpoint(
                restore_checkpoint_path=fit_config.checkpointer.restore_checkpoint_path,
                optimizer=fit_config.optimizer.method,
            )

            if not isinstance(state, allowed_states):
                raise ValueError(f"The type of the restored checkpoint must be within {allowed_states}. "
                                 f"However, {fit_config.checkpointer.restore_checkpoint_path} pointed to a state "
                                 f"with type {type(state)}.")

            if isinstance(state, ADVIState):
                means, log_stds = nested_unpair(
                    d=state.params.unfreeze(),
                    key_paths=self.posterior_approximator.which_params,
                    labels=("mean", "log_std")
                ) if self.posterior_approximator.which_params is not None else \
                    [FrozenDict({k: dict(params=v["params"][s]) for k, v in state.params.items()}) for s in ["mean", "log_std"]]
                means, log_stds = FrozenDict(means), FrozenDict(log_stds)
                state = state.replace(params=means)
                del means
        elif map_fit_config is not None:
            map_posterior = MAPPosterior(
                self.joint, posterior_approximator=MAPPosteriorApproximator()
            )
            map_posterior.rng = self.rng
            logging.info("Do a preliminary run of MAP.")
            status["map"] = map_posterior.fit(
                rng=self.rng.get(),
                train_data_loader=train_data_loader,
                val_data_loader=val_data_loader,
                fit_config=map_fit_config,
                **kwargs,
            )
            logging.info("Preliminary run with MAP completed.")
            state = map_posterior.state.get()
        elif self.posterior_approximator.which_params is not None:
            logging.warning(
                f"ADVI will run exclusively over the parameters specified in `which_params`."
                f"All the other parameters will be initialized randomly. "
                f"If you want to train the other parameters too, "
                f"and yet run ADVI only on the parameters in `which_params`, "
                f"you can either restore an existing checkpoint, or set `map_fit_config`. "
                f"In the latter case, a pre-run of MAP will run before ADVI.")

        state, n_train_data, n_val_data = self._init(
            train_data_loader, val_data_loader, state
        )

        return state, status, n_train_data, n_val_data

    def _get_unravel(self, params, log_stds: Optional[Array] = None):
        if self.posterior_approximator.which_params is None:
            rav, unravel = ravel_pytree(params)
            rav_log_stds = ravel_pytree(log_stds)[0] if log_stds is not None else None
            indices = None
        else:
            def unravel_fn(_params, _path):
                return ravel_pytree(nested_get(_params, _path))
            rav, unravel, indices, rav_log_stds = [], [], [], []
            for path in self.posterior_approximator.which_params:
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
