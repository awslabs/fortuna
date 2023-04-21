from __future__ import annotations

import abc
from typing import Any, Callable, Dict, Optional, Tuple, Union, List

import jax.numpy as jnp
from flax.core import FrozenDict
from jax import random, vmap
from jax._src.prng import PRNGKeyArray
from jax.tree_util import tree_map
from optax._src.base import PyTree

from fortuna.distribution.base import Distribution
from fortuna.prob_model.posterior.posterior_trainer import PosteriorTrainerABC
from fortuna.prob_model.posterior.state import PosteriorState
from fortuna.prob_model.fit_config.callbacks import Callback
from fortuna.typing import Array, Batch, CalibMutable, CalibParams, Params, Mutable


class NormalizingFlowTrainer(PosteriorTrainerABC):
    @abc.abstractmethod
    def __str__(self):
        pass

    def __init__(self, *, base: Distribution, architecture: object, **kwargs):
        """
        This is a Normalizing Flow. It is able to transform a `base` distribution into a target one via an invertible
        `architecture`.

        :param base: object
            Base distribution. It must include the following methods:
                - sample(rng: jnp.array, n_samples: int) -> jnp.array
                    It draws `n_samples` samples from the distribution.
                - log_joint_prob(x: jnp.array) -> float
                    It evaluates the log-probability density function at `x`.
            Common distribution are already available in `prob.distribution`.
        :param architecture: object
            Invertible architecture. It must include the following methods:
                - forward(params: jnp.array, u: jnp.array) ->  jnp.array
                    It applies on `u` the transformation parametrized by `params`.
                - backward(params: jnp.array, v: jnp.array) ->  jnp.array
                    It applies on `v` the inverse transformation parametrized by `params`.
                - init_params(rng: jnp.array)
                    It initializes the parameters of the transformation.
        """
        super(NormalizingFlowTrainer, self).__init__(**kwargs)
        # base distribution
        self.sample_base = base.sample
        self.base_log_joint_prob = base.log_joint_prob

        # architecture
        self.forward = architecture.forward
        self.backward = architecture.backward
        self.init_params = architecture.init_params

    def sample_forward(self, rng: jnp.array, params: any, n_samples: int) -> tuple:
        """
        Sample from the push-forward distribution.

        :param rng: jnp.array
            Random number generator.
        :param params: any
            Transformation parameters.
        :param n_samples:
            Number of samples to draw.

        :return: tuple
            forward_samples: jnp.array
                Samples.
            ldj: jnp.array
                Log-determinant of the Jacobians.
        """
        return self.forward(params, self.sample_base(rng, n_samples))

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
        v, ldj = self.sample_forward(key, tuple(params.values()), kwargs["n_samples"])
        neg_logp, aux = vmap(
            lambda _params: loss_fun(
                unravel(_params),
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
        """
        Perform training batch metrics_names computation and save checkpoint if needed.

        :param current_epoch: int
            Current epoch.
        :param state: TrainState
            The training state.
        :param aux: Dict[str, Any]
            The dictionary obtained from a call to `training_loss_step()`. It must have a key name `outputs` which
            contains the model's prediction for the given `batch`.
        :param batch: Batch
            The input data and the targets.
        :param metrics: Optional[Tuple[Callable[[jnp.ndarray, Array], float], ...]]
            A tuple of metrics.
        :param callbacks: Optional[List[Callback]]
            A list of user-defined callbacks that runs sequentially.
        :param kwargs: FrozenDict[str, Any]
            Any other extra argument. They have to be explicitly passed within a dictionary and cannot be provided as
            named arguments due to `jax.jit`.

        :return: Dict[str, jnp.ndarray]
            A dictionary containing `metrics_names` values on the given input batch.
        """
        if (
            self.save_checkpoint_dir
            and self.save_every_n_steps
            and current_epoch % self.save_every_n_steps
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
        v, ldj = self.sample_forward(
            key, tuple(state.params.values()), kwargs["n_samples"]
        )
        neg_logp, aux = vmap(
            lambda _params: loss_fun(
                unravel(_params),
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
