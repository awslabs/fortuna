import copy
import logging

import importlib

import jax
from flax.core import unfreeze
from flax.traverse_util import flatten_dict
from transformers import PretrainedConfig, AutoConfig
from transformers.models.auto.auto_factory import _LazyAutoMapping
from transformers.models.auto.configuration_auto import model_type_to_module_name

logger = logging.getLogger(__name__)


def _get_model_class(config, model_mapping):
    supported_models = model_mapping[type(config)]
    if not isinstance(supported_models, (list, tuple)):
        return supported_models

    name_to_model = {model.__name__: model for model in supported_models}
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in name_to_model:
            return name_to_model[arch]
        elif f"TF{arch}" in name_to_model:
            return name_to_model[f"TF{arch}"]
        elif f"Flax{arch}" in name_to_model:
            return name_to_model[f"Flax{arch}"]

    # If not architecture is set in the config or match the supported models, the first element of the tuple is the
    # defaults.
    return supported_models[0]


class _BaseAutoSNGPModelClass:
    # Base class for auto sngp models.
    _model_mapping = None

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)`"
        )

    @classmethod
    def from_config(cls, config, **kwargs):
        raise EnvironmentError(
            f"{cls.__name__} is designed to be instantiated "
            f"using the `{cls.__name__}.from_pretrained(pretrained_model_name_or_path)`."
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        trust_remote_code = kwargs.pop("trust_remote_code", False)
        kwargs["_from_auto"] = True
        hub_kwargs_names = [
            "cache_dir",
            "force_download",
            "local_files_only",
            "proxies",
            "resume_download",
            "revision",
            "subfolder",
            "use_auth_token",
        ]
        hub_kwargs = {
            name: kwargs.pop(name) for name in hub_kwargs_names if name in kwargs
        }
        if not isinstance(config, PretrainedConfig):
            kwargs_copy = copy.deepcopy(kwargs)
            # ensure not to pollute the config object with torch_dtype="auto" - since it's
            # meaningless in the context of the config object - torch.dtype values are acceptable
            if kwargs_copy.get("torch_dtype", None) == "auto":
                _ = kwargs_copy.pop("torch_dtype")

            config, kwargs = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                return_unused_kwargs=True,
                trust_remote_code=trust_remote_code,
                **hub_kwargs,
                **kwargs_copy,
            )
        if type(config) in cls._model_mapping.keys():
            model_class = _get_model_class(config, cls._model_mapping)
            model = model_class.from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                config=config,
                **hub_kwargs,
                **kwargs,
            )
            spectral_stats_params = model.params.pop("spectral_stats", None)
            batch_stats_params = model.params.pop("batch_stats", None)
            restructured_params = {"params": model.params}
            if spectral_stats_params:
                restructured_params.update({"spectral_stats": spectral_stats_params})
            if batch_stats_params:
                restructured_params.update({"batch_stats": batch_stats_params})
            # update shape of the parameters
            params_shape_tree = jax.eval_shape(
                lambda params: params, restructured_params
            )
            model._params_shape_tree = params_shape_tree
            # update required_params
            model._required_params = set(
                flatten_dict(unfreeze(params_shape_tree)).keys()
            )
            # update params
            model.params = restructured_params
            return model
        raise ValueError(
            f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\n"
            f"At the moment, the model types that supports SNGP training are {', '.join(c.__name__ for c in cls._model_mapping.keys())}."
        )


def getattribute_from_module(module, attr):
    if attr is None:
        return None
    if isinstance(attr, tuple):
        return tuple(getattribute_from_module(module, a) for a in attr)
    if hasattr(module, attr):
        return getattr(module, attr)
    raise ValueError(f"Could not find {attr} in {module}!")


class _SNGPLazyAutoMapping(_LazyAutoMapping):
    def _load_attr_from_module(self, model_type, attr):
        module_name = model_type_to_module_name(model_type)
        if module_name not in self._modules:
            self._modules[module_name] = importlib.import_module(
                f".modeling_flax_{module_name}",
                "fortuna.prob_model.posterior.sngp.transformers.models",
            )
        return getattribute_from_module(self._modules[module_name], attr)
