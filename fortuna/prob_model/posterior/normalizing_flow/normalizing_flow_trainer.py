from __future__ import annotations

import abc
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import jax.numpy as jnp
from flax.core import FrozenDict
from jax import (
    random,
    vmap,
)
from jax._src.prng import PRNGKeyArray
from jax.tree_util import tree_map
from optax._src.base import PyTree

from fortuna.distribution.base import Distribution
from fortuna.prob_model.posterior.posterior_trainer import PosteriorTrainerABC
from fortuna.prob_model.posterior.state import PosteriorState
from fortuna.training.callback import Callback
from fortuna.typing import (
    Array,
    Batch,
    CalibMutable,
    CalibParams,
    Mutable,
    Params,
)
from fortuna.utils.nested_dicts import nested_set
from fortuna.utils.strings import encode_tuple_of_lists_of_strings_to_numpy


class NormalizingFlowTrainer(PosteriorTrainerABC):
    @abc.abstractmethod
    def __str__(self):
        pass

    def sample_forward(self, rng: jnp.array, params: any, n_samples: int) -> tuple:
        """
        Sample from the push-forward distribution.

        Parameters
        ----------
        rng: PRNGKeyArray
            Random number generator.
        params: Params
            Transformation parameters.
        n_samples: int
            Number of samples to draw.

        Returns
        -------
        tuple
            forward_samples: jnp.ndarray
                Samples.
            ldj: jnp.ndarray
                Log-determinant of the Jacobians.
        """
        return self.forward(params, self.sample_base(rng, n_samples))

    def __init__(
        self,
        *,
        base: Distribution,
        architecture: object,
        which_params: Optional[Tuple[List[str]]],
        all_params: Optional[Params] = None,
        indices: Optional[List[int]] = None,
        unravel: Union[List[Callable], Callable],
        **kwargs,
    ):
        super(NormalizingFlowTrainer, self).__init__(**kwargs)
        # base distribution
        self.sample_base = base.sample
        self.base_log_joint_prob = base.log_joint_prob

        # architecture
        self.forward = architecture.forward
        self.backward = architecture.backward
        self.init_params = architecture.init_params

        # Normalizing flows on subsets of the model parameters
        self._which_params = which_params
        self._encoded_which_params = encode_tuple_of_lists_of_strings_to_numpy(
            which_params
        )
        self._all_params = all_params
        self._indices = indices
        self._unravel = unravel

    def training_loss_step(
        self,
        loss_fun: Callable[[Any], Union[float, Tuple[float, dict]]],
        params: Params,
        batch: Batch,
        mutable: Mutable,
        rng: PRNGKeyArray,
        n_data: int,
        unravel: Optional[Callable[[any], PyTree]] = None,
        calib_params: Optional[CalibParams] = None,
        calib_mutable: Optional[CalibMutable] = None,
        kwargs: FrozenDict[str, Any] = FrozenDict(),
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        return_aux = ["outputs"]
        if mutable is not None:
            return_aux += ["mutable"]
        rng, key = random.split(rng)
        v, ldj = self.sample_forward(key, params, kwargs["n_samples"])
        neg_logp, aux = vmap(
            lambda _rav_params: loss_fun(
                self._get_params_from_rav(_rav_params, unravel),
                batch,
                n_data=n_data,
                mutable=mutable,
                return_aux=return_aux,
                train=True,
                rng=rng,
                calib_params=calib_params,
                calib_mutable=calib_mutable,
            )
        )(v)
        return (
            jnp.mean(neg_logp) - jnp.mean(ldj),
            {
                "outputs": aux.get("outputs"),
                "mutable": tree_map(lambda x: x[-1], aux.get("mutable")),
                "logging_kwargs": dict(),
            },
        )

    def training_step_end(
        self,
        current_epoch: int,
        state: PosteriorState,
        aux: Dict[str, Any],
        batch: Batch,
        metrics: Optional[Tuple[Callable[[jnp.ndarray, Array], float], ...]] = None,
        callbacks: Optional[List[Callback]] = None,
        kwargs: FrozenDict[str, Any] = FrozenDict(),
    ) -> Tuple[PosteriorState, Dict[str, jnp.ndarray]]:
        if (
            self.save_checkpoint_dir is not None
            and self.save_every_n_steps is not None
            and self.save_every_n_steps > 0
            and self._global_training_step >= self.save_every_n_steps
            and self._global_training_step % self.save_every_n_steps == 0
        ):
            self.save_checkpoint(
                state, self.save_checkpoint_dir, keep=self.keep_top_n_checkpoints
            )
        training_losses_and_metrics = {"loss": aux["loss"]}

        if aux["logging_kwargs"] is not None:
            for k, v in aux["logging_kwargs"].items():
                training_losses_and_metrics[k] = v

        if not self.disable_training_metrics_computation and metrics is not None:
            preds = self.predict_fn(aux["outputs"])
            if self.multi_device:
                training_batch_metrics = vmap(
                    lambda p: self.compute_metrics(
                        p,
                        batch[1].reshape(
                            (batch[1].shape[0] * batch[1].shape[1],)
                            + batch[1].shape[2:]
                        ),
                        metrics,
                    )
                )(
                    preds.reshape(
                        (
                            preds.shape[1],
                            preds.shape[0] * preds.shape[2],
                        )
                        + preds.shape[3:]
                    )
                )
            else:
                training_batch_metrics = vmap(
                    lambda p: self.compute_metrics(p, batch[1], metrics)
                )(preds)
            training_batch_metrics = tree_map(
                lambda m: m.mean(), training_batch_metrics
            )
            for k, v in training_batch_metrics.items():
                training_losses_and_metrics[k] = v
        state = self._callback_loop(state, callbacks, "training_step_end")
        return state, training_losses_and_metrics

    def validation_step(
        self,
        state: PosteriorState,
        batch: Batch,
        loss_fun: Callable[[Any], Union[float, Tuple[float, dict]]],
        rng: PRNGKeyArray,
        n_data: int,
        metrics: Optional[Tuple[Callable[[jnp.ndarray, Array], float]], ...] = None,
        unravel: Optional[Callable[[any], PyTree]] = None,
        kwargs: FrozenDict[str, Any] = FrozenDict(),
    ) -> Dict[str, jnp.ndarray]:
        rng, key = random.split(rng)
        v, ldj = self.sample_forward(key, state.params, kwargs["n_samples"])
        neg_logp, aux = vmap(
            lambda _rav_params: loss_fun(
                self._get_params_from_rav(_rav_params, unravel),
                batch,
                n_data=n_data,
                mutable=state.mutable,
                return_aux=["outputs"],
                train=False,
                rng=rng,
                calib_params=state.calib_params,
                calib_mutable=state.calib_mutable,
            )
        )(v)
        loss = jnp.mean(neg_logp) - jnp.mean(ldj)
        if metrics is not None:
            preds = self.predict_fn(aux["outputs"])
            val_metrics = vmap(lambda p: self.compute_metrics(p, batch[1], metrics))(
                preds
            )
            val_metrics = tree_map(lambda m: m.mean(), val_metrics)
            return {
                "val_loss": loss,
                **{f"val_{m}": v for m, v in val_metrics.items()},
            }
        return dict(val_loss=loss)

    def _get_params_from_rav(self, _rav, unravel) -> Params:
        if self._which_params is None:
            return unravel(_rav)
        return FrozenDict(
            nested_set(
                d=self._all_params.unfreeze(),
                key_paths=self._which_params,
                objs=tuple(
                    [
                        _unravel(_rav[self._indices[i] : self._indices[i + 1]])
                        for i, _unravel in enumerate(unravel)
                    ]
                ),
            )
        )
