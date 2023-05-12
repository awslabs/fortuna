from typing import Dict, Optional, Union, Tuple, Mapping

import jax
from flax import linen as nn
from flax.core import FrozenDict
from flax.training.checkpoints import PyTree
from jax import numpy as jnp, random
from jax._src.prng import PRNGKeyArray

from fortuna.model.model_manager.classification import (
    ClassificationModelManager,
    SNGPClassificationModelManagerMixin,
)
from fortuna.typing import Params, Array, Mutable
from fortuna.utils.data import get_inputs_from_shape
from fortuna.utils.nested_dicts import nested_update


class HuggingFaceClassificationModelManager(ClassificationModelManager):
    def apply(
        self,
        params: Params,
        inputs: Dict[str, Array],
        mutable: Optional[Mutable] = None,
        train: bool = False,
        rng: Optional[PRNGKeyArray] = None,
        **kwargs,
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, PyTree]]:
        # setup dropout key
        if rng is not None:
            rng, dropout_rng = random.split(rng, 2)
        else:
            dropout_rng = None

        model_kwargs = {}
        if mutable is not None:
            model_kwargs = {"mutable": mutable}
        outputs = self.model(
            **inputs,
            params=params["model"]["params"],
            dropout_rng=dropout_rng,
            train=train,
            output_attentions=kwargs.get("output_attentions"),
            output_hidden_states=kwargs.get("output_hidden_states"),
            return_dict=kwargs.get("return_dict"),
            **model_kwargs,
        )
        if train and mutable:
            outputs, mutable = outputs
        if hasattr(outputs, "logits"):
            outputs = outputs.logits
        if train and mutable:
            return outputs, {"mutable": FrozenDict({"model": mutable})}
        return outputs

    def init(
        self, input_shape: Tuple[int, ...], rng: Optional[PRNGKeyArray] = None, **kwargs
    ) -> Dict[str, Mapping]:
        assert self.model._is_initialized, (
            "At the moment Fortuna supports models from Hugging Face that are loaded via "
            "`from_pretrained` method, which also takes care of model initialization."
        )
        return {"model": {"params": self.model.params}}


class SNGPHuggingFaceClassificationModelManager(
    SNGPClassificationModelManagerMixin, HuggingFaceClassificationModelManager
):
    def __init__(self, model: nn.Module, *args, **kwargs):
        super(SNGPHuggingFaceClassificationModelManager, self).__init__(
            model, *args, **kwargs
        )

    def init(
        self, input_shape: Tuple[int, ...], rng: Optional[PRNGKeyArray] = None, **kwargs
    ) -> Dict[str, FrozenDict]:
        if rng is None:
            rng = self.rng.get()
        assert self.model._is_initialized, (
            "At the moment Fortuna supports models from Hugging Face that are loaded via "
            "`from_pretrained` method, which also takes care of model initialization."
        )
        output_shape = jax.eval_shape(
            self.model, **get_inputs_from_shape(input_shape)
        ).shape
        if len(output_shape[1:]) > 1:  # drop batch size
            raise ValueError(
                f"The output shape for the given model is {output_shape}.\n"
                f"In order to use SNGP the output shape of the provide model has to be of shape"
                f"(batch_size, n_features)."
            )
        rng, params_key, dropout_key = random.split(rng, 3)
        rngs = {"params": params_key, "dropout": dropout_key}
        gp_params = self._gp_output_model.init(rngs, jnp.zeros(output_shape), **kwargs)
        params = nested_update(self.model.params, gp_params.unfreeze())
        return dict(model=FrozenDict(params))
