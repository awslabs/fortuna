from __future__ import annotations

import logging
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from flax.core import FrozenDict
from jax._src.prng import PRNGKeyArray
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import numpy as np

from fortuna.data.loader import (
    DataLoader,
    InputsLoader,
)
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
from fortuna.typing import (
    AnyKey,
    Array,
    OptaxOptimizer,
    Params,
    Path,
    Status,
)
from fortuna.utils.builtins import get_dynamic_scale_instance_from_model_dtype
from fortuna.utils.device import select_trainer_given_devices
from fortuna.utils.freeze import get_trainable_paths
from fortuna.utils.nested_dicts import (
    nested_get,
    nested_set,
    nested_unpair,
)
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
        self._base = None
        self._architecture = None
        self._unravel = None
        self._indices = None

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

        rav, self._unravel, sub_unravel, rav_log_stds = self._get_unravel(
            params=state.params, log_stds=log_stds, which_params=which_params
        )

        size_rav = len(rav)
        self._base, self._architecture = self._get_base_and_architecture(size_rav)

        trainer_cls = select_trainer_given_devices(
            devices=fit_config.processor.devices,
            base_trainer_cls=ADVITrainer,
            jitted_trainer_cls=JittedADVITrainer,
            multi_device_trainer_cls=MultiDeviceADVITrainer,
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
            freeze_fun=fit_config.optimizer.freeze_fun,
            base=self._base,
            architecture=self._architecture,
            which_params=which_params,
            unravel=self._unravel,
            sub_unravel=sub_unravel,
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
            validation_dataset_size=(
                val_data_loader.size if val_data_loader is not None else None
            ),
            verbose=fit_config.monitor.verbose,
            unravel=self._unravel,
            n_samples=self.posterior_approximator.n_loss_samples,
            callbacks=fit_config.callbacks,
            max_grad_norm=fit_config.hyperparameters.max_grad_norm,
            gradient_accumulation_steps=fit_config.hyperparameters.gradient_accumulation_steps,
        )

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

    def load_state(self, checkpoint_path: Path) -> None:
        """
        Load the state of the posterior distribution from a checkpoint path. The checkpoint must be
        compatible with the current probabilistic model.

        Parameters
        ----------
        checkpoint_path: Path
            Path to checkpoint file or directory to restore.
        """
        try:
            self.restore_checkpoint(checkpoint_path)
        except ValueError:
            raise ValueError(
                f"No checkpoint was found in `checkpoint_path={checkpoint_path}`."
            )
        self.state = PosteriorStateRepository(checkpoint_dir=checkpoint_path)

        state = self.state.get()
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
        _base, _architecture = self._get_base_and_architecture(n_params)
        _unravel = self._get_unravel(
            FrozenDict(
                nested_unpair(
                    d=state.params.unfreeze(),
                    key_paths=which_params,
                    labels=("mean", "log_std"),
                )[0]
                if which_params
                else {
                    k: dict(params=v["params"]["mean"]) for k, v in state.params.items()
                }
            ),
            which_params=which_params,
        )[1]

        self._base = _base
        self._architecture = _architecture
        self._unravel = _unravel

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

        _base = self._base
        _architecture = self._architecture
        _unravel = self._unravel

        if state._encoded_which_params is None:
            means = _unravel(
                _architecture.forward(
                    {
                        s: ravel_pytree(
                            {k: v["params"][s] for k, v in state.params.items()}
                        )[0]
                        for s in ["mean", "log_std"]
                    },
                    _base.sample(rng),
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
            rav_params = _architecture.forward(params=rav_params, u=_base.sample(rng))[
                0
            ][0]

            means = _unravel(rav_params)

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
            dynamic_scale=get_dynamic_scale_instance_from_model_dtype(
                getattr(self.joint.likelihood.model_manager.model, "dtype")
                if hasattr(self.joint.likelihood.model_manager.model, "dtype")
                else None
            ),
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
            dynamic_scale=get_dynamic_scale_instance_from_model_dtype(
                getattr(self.joint.likelihood.model_manager.model, "dtype")
                if hasattr(self.joint.likelihood.model_manager.model, "dtype")
                else None
            ),
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
            sub_unravel = None
        else:
            rav, sub_unravel = ravel_pytree(
                tuple([nested_get(params, path) for path in which_params])
            )

            def unravel(_rav):
                return FrozenDict(
                    nested_set(
                        d=params.unfreeze(),
                        key_paths=which_params,
                        objs=sub_unravel(_rav),
                    )
                )

            if log_stds is not None:
                rav_log_stds = ravel_pytree(
                    nested_set(
                        d={},
                        key_paths=which_params,
                        objs=tuple(
                            [nested_get(log_stds, path) for path in which_params]
                        ),
                        allow_nonexistent=True,
                    )
                )[0]
            else:
                rav_log_stds = None

        return rav, unravel, sub_unravel, rav_log_stds
