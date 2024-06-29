from typing import (
    Dict,
    Mapping,
    Optional,
    Tuple,
    Union,
)

from flax import linen as nn
from flax.core import FrozenDict
from flax.training.checkpoints import PyTree
import jax
from jax import (
    numpy as jnp,
    random,
)

from fortuna.model.model_manager.classification import (
    ClassificationModelManager,
    SNGPClassificationModelManagerMixin,
)
from fortuna.model_editor.base import ModelEditor
from fortuna.typing import (
    Array,
    Mutable,
    Params,
)
from fortuna.utils.data import get_inputs_from_shape
from fortuna.utils.nested_dicts import nested_update


class HuggingFaceClassificationModelManager(ClassificationModelManager):
    def apply(
        self,
        params: Params,
        inputs: Dict[str, Array],
        mutable: Optional[Mutable] = None,
        train: bool = False,
        rng: Optional[jax.Array] = None,
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

        has_aux = train and mutable

        def apply_fn(p, x):
            _outputs = self.model(
                **x,
                params=p,
                dropout_rng=dropout_rng,
                train=train,
                output_attentions=kwargs.get("output_attentions"),
                output_hidden_states=kwargs.get("output_hidden_states"),
                return_dict=kwargs.get("return_dict"),
                **model_kwargs,
            )
            if hasattr(_outputs, "logits"):
                _outputs = _outputs.logits

            if isinstance(_outputs, tuple) and not has_aux:
                _outputs = _outputs[0]
            return _outputs

        if self.model_editor is not None:
            outputs = self.model_editor.apply(
                params["model_editor"],
                apply_fn=apply_fn,
                model_params=params["model"]["params"],
                x=inputs,
                has_aux=has_aux,
            )
        else:
            outputs = apply_fn(params["model"]["params"], inputs)

        if has_aux:
            outputs, mutable = outputs
            return outputs, {"mutable": FrozenDict({"model": mutable})}
        return outputs

    def init(
        self, input_shape: Tuple[int, ...], rng: Optional[jax.Array] = None, **kwargs
    ) -> Dict[str, Mapping]:
        assert self.model._is_initialized, (
            "At the moment Fortuna supports models from Hugging Face that are loaded via "
            "`from_pretrained` method, which also takes care of model initialization."
        )
        params = {"model": {"params": self.model.params}}
        if self.model_editor is not None:
            if rng is None:
                rng = self.rng.get()
            rng, params_key, dropout_key = random.split(rng, 3)
            rngs = {"params": params_key, "dropout": dropout_key}

            def apply_fn(p, x):
                _outputs = self.model(**x, params=p)
                if hasattr(_outputs, "logits"):
                    _outputs = _outputs.logits
                return _outputs

            params.update(
                dict(
                    model_editor=self.model_editor.init(
                        rngs,
                        apply_fn=apply_fn,
                        model_params=FrozenDict(params["model"]["params"]),
                        x=get_inputs_from_shape(input_shape),
                        has_aux=False,
                    )
                )
            )
        return params


class SNGPHuggingFaceClassificationModelManager(
    SNGPClassificationModelManagerMixin, HuggingFaceClassificationModelManager
):
    def __init__(
        self,
        model: nn.Module,
        model_editor: Optional[ModelEditor] = None,
        *args,
        **kwargs,
    ):
        super(SNGPHuggingFaceClassificationModelManager, self).__init__(
            model, model_editor=model_editor, *args, **kwargs
        )

    def init(
        self, input_shape: Tuple[int, ...], rng: Optional[jax.Array] = None, **kwargs
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
        params = dict(model=nested_update(self.model.params, gp_params.unfreeze()))

        if self.model_editor is not None:
            rng, params_key, dropout_key = random.split(rng, 3)
            rngs = {"params": params_key, "dropout": dropout_key}

            mutable = dict(
                model=FrozenDict(
                    {k: v for k, v in params["model"].items() if k != "params"}
                )
            )

            def apply_fn(p, x):
                _outputs = self.model(**x, params=p, mutable=mutable)
                return _outputs

            params.update(
                dict(
                    model_editor=self.model_editor.init(
                        rngs,
                        apply_fn=apply_fn,
                        model_params=FrozenDict(params["model"]["params"]),
                        x=get_inputs_from_shape(input_shape),
                        has_aux=False,
                    )
                )
            )
        return params
