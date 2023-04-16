from typing import Dict, Optional, Tuple, Union

import flax.linen as nn
import jax.numpy as jnp
from flax.core import FrozenDict
from flax.training.checkpoints import PyTree
from jax import random
from jax._src.prng import PRNGKeyArray

from fortuna.model.model_manager.base import ModelManager
from fortuna.typing import Array, Mutable, Params

__all__ = ["ClassificationModelManager", "SNGPModelManager"]


class ClassificationModelManager(ModelManager):
    def __init__(self, model: nn.Module):
        r"""
        Classification model manager class. It orchestrates the forward pass of the model in the probabilistic model.

        Parameters
        ----------
        model : nn.Module
            A model describing the deterministic relation between inputs and outputs. The outputs must correspond to
            the logits of a softmax probability vector. The output dimension must be the same as the number of classes.
            Let :math:`x` be input variables and :math:`w` the random model parameters. Then the model is described by
            a function :math:`f(w, x)`, where each component of :math:`f` corresponds to one of the classes.
        """
        self.model = model

    def apply(
        self,
        params: Params,
        inputs: Array,
        mutable: Optional[Mutable] = None,
        train: bool = False,
        rng: Optional[PRNGKeyArray] = None,
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, PyTree]]:
        if mutable is None:
            mutable = False
        variables = params["model"].unfreeze()

        # setup dropout key
        if rng is not None:
            rng, dropout_rng = random.split(rng, 2)
            rngs = {"dropout": dropout_rng}
        else:
            rngs = None

        if mutable:
            mutable_variables = mutable["model"].unfreeze()
            variables.update(mutable_variables)
            mutable = list(mutable_variables.keys())
        if train and mutable:
            outputs, mutable = self.model.apply(
                variables, inputs, mutable=mutable, train=train, rngs=rngs
            )
            if isinstance(outputs, tuple):
                # supports for sngp
                outputs = outputs[0]
            return outputs, {"mutable": FrozenDict({"model": mutable})}
        else:
            return self.model.apply(
                variables, inputs, train=train, mutable=False, rngs=rngs
            )

    def init(
        self, input_shape: Tuple[int, ...], rng: Optional[PRNGKeyArray] = None, **kwargs
    ) -> Dict[str, FrozenDict]:
        if rng is None:
            rng = self.rng.get()
        rng, params_key, dropout_key = random.split(rng, 3)
        rngs = {"params": params_key, "dropout": dropout_key}
        return dict(
            model=self.model.init(rngs, jnp.zeros((1,) + input_shape), **kwargs)
        )


class SNGPModelManager(ClassificationModelManager):
    def __init__(self, *args, mean_field_factor: float = jnp.pi/8, likelihood: str = 'logistic', **kwargs):
        """
        Classification model manager for SNGP models.

        Parameters
        ----------
        mean_field_factor: float
            The scale factor for mean-field approximation, used to adjust (at inference time) the influence of
            posterior variance in posterior mean approximation.
        likelihood: str
            Likelihood for integration in Gaussian-approximated latent posterior.
        """
        super(SNGPModelManager, self).__init__(*args, **kwargs)
        self.mean_field_factor = mean_field_factor
        self.likelihood = likelihood

    def _mean_field_logits(self, logits: Array, covmat: Optional[Array] = None) -> Array:
        """
        Adjust the model logits so its softmax approximates the posterior mean
        ([Zhiyun L. et al., 2020](https://arxiv.org/abs/2006.07584)).

        Parameters
        ----------
          logits: Array
            A float tensor of shape (batch_size, num_classes).
          covmat: Array
            A float tensor of shape (batch_size, batch_size). If None then it
            assumes the covmat is an identity matrix.

        Returns
        ----------
        Array
            True or False if `pred` has a constant boolean value, None otherwise.
        """
        # Implementation taken from https://github.com/google/edward2/blob/520e28285e905e0021e49b52b982ee5ea170071c/edward2/tensorflow/layers/utils.py#L379
        if self.likelihood not in ('logistic', 'binary_logistic', 'poisson'):
            raise ValueError(
                f'Likelihood" must be one of (\'logistic\', \'binary_logistic\', \'poisson\'), got {self.likelihood}.'
            )

        if self.mean_field_factor < 0:
            return logits

        # Compute standard deviation.
        if covmat is None:
            variances = 1.
        else:
            variances = jnp.diagonal(covmat)

        # Compute scaling coefficient for mean-field approximation.
        if self.likelihood == 'poisson':
            logits_scale = jnp.exp(- variances * self.mean_field_factor / 2.)
        else:
            logits_scale = jnp.sqrt(1. + variances * self.mean_field_factor)

        # Cast logits_scale to compatible dimension.
        if len(logits.shape) > 1:
            logits_scale = jnp.expand_dims(logits_scale, axis=-1)

        return logits / logits_scale

    def apply(
        self,
        params: Params,
        inputs: Array,
        mutable: Optional[Mutable] = None,
        train: bool = False,
        rng: Optional[PRNGKeyArray] = None,
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, PyTree]]:
        outputs = super(SNGPModelManager, self).apply(params, inputs, mutable, train, rng)
        if train and mutable:
            return outputs
        else:
            logits, covmat = outputs
            return self._mean_field_logits(logits, covmat)