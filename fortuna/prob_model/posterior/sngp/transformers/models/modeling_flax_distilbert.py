import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core import FrozenDict, unfreeze, freeze
from flax.traverse_util import flatten_dict, unflatten_dict
from transformers import DistilBertConfig
from transformers.modeling_flax_utils import ACT2FN
from transformers.models.distilbert.modeling_flax_distilbert import (
    FlaxDistilBertModule,
    FlaxDistilBertPreTrainedModel,
)
from typing import Tuple, Optional

from fortuna.model.utils.spectral_norm import WithSpectralNorm


class FlaxSNGPDistilBertExtractorForSequenceClassificationModule(
    nn.Module, WithSpectralNorm
):
    config: DistilBertConfig = None
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.distilbert = FlaxDistilBertModule(config=self.config, dtype=self.dtype)
        self.pre_classifier = self.spectral_norm(nn.Dense)(
            self.config.dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(
                stddev=self.config.initializer_range
            ),
        )
        self.dropout = nn.Dropout(rate=self.config.seq_classif_dropout)

    def __call__(
        self,
        input_ids,
        attention_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Model
        distilbert_output = self.distilbert(
            input_ids,
            attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = ACT2FN["relu"](pooled_output)
        pooled_output = self.dropout(pooled_output, deterministic=deterministic)

        if not return_dict:
            return (pooled_output,) + distilbert_output[1:]

        return pooled_output


class FlaxSNGPDistilBertExtractorForSequenceClassification(
    FlaxDistilBertPreTrainedModel
):
    module_class = FlaxSNGPDistilBertExtractorForSequenceClassificationModule

    def init_weights(
        self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None
    ) -> FrozenDict:
        # init input tensors
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        random_params = self.module.init(
            rngs, input_ids, attention_mask, return_dict=False
        )
        random_params = FrozenDict(
            {
                "spectral_stats": random_params["spectral_stats"].unfreeze(),
                **random_params["params"].unfreeze(),
            }
        )
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        head_mask=None,
        params: dict = None,
        mutable: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )

        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        variables = {"params": params} if params is not None else self.params
        if mutable:
            mutable_variables = mutable["model"].unfreeze()
            variables.update(mutable_variables)
            mutable = list(mutable_variables.keys())

        return self.module.apply(
            variables,
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            not train,
            output_attentions,
            output_hidden_states,
            return_dict,
            mutable=mutable if train else False,
            rngs=rngs if train else None,
        )
