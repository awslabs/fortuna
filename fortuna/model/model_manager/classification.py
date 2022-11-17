from typing import Dict, Optional, Tuple, Union

import flax.linen as nn
import jax.numpy as jnp
from flax.core import FrozenDict
from flax.training.checkpoints import PyTree
from fortuna.model.model_manager.base import ModelManager
from fortuna.typing import Array, Mutable, Params
from jax import random
from jax._src.prng import PRNGKeyArray


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
