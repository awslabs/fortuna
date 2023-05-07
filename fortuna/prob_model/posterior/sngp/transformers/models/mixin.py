# import jax
# import jax.numpy as jnp
#
#
# class FromPreTrainedSNGP:
#
#     @classmethod
#     def from_pretrained(
#         cls,
#         pretrained_model_name_or_path: Union[str, os.PathLike],
#         dtype: jnp.dtype = jnp.float32,
#         *model_args,
#         **kwargs,
#     ):
#         r"""
#         Instantiate a pretrained flax model from a pre-trained model configuration.
#
#         The warning *Weights from XXX not initialized from pretrained model* means that the weights of XXX do not come
#         pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
#         task.
#
#         The warning *Weights from XXX not used in YYY* means that the layer XXX is not used by YYY, therefore those
#         weights are discarded.
#
#         Parameters:
#             pretrained_model_name_or_path (`str` or `os.PathLike`):
#                 Can be either:
#
#                     - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
#                       Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
#                       user or organization name, like `dbmdz/bert-base-german-cased`.
#                     - A path to a *directory* containing model weights saved using
#                       [`~FlaxPreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
#                     - A path or url to a *pt index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In this case,
#                       `from_pt` should be set to `True`.
#             dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
#                 The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and
#                 `jax.numpy.bfloat16` (on TPUs).
#
#                 This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
#                 specified all the computation will be performed with the given `dtype`.
#
#                 **Note that this only specifies the dtype of the computation and does not influence the dtype of model
#                 parameters.**
#
#                 If you wish to change the dtype of the model parameters, see [`~FlaxPreTrainedModel.to_fp16`] and
#                 [`~FlaxPreTrainedModel.to_bf16`].
#             model_args (sequence of positional arguments, *optional*):
#                 All remaining positional arguments will be passed to the underlying model's `__init__` method.
#             config (`Union[PretrainedConfig, str, os.PathLike]`, *optional*):
#                 Can be either:
#
#                     - an instance of a class derived from [`PretrainedConfig`],
#                     - a string or path valid as input to [`~PretrainedConfig.from_pretrained`].
#
#                 Configuration for the model to use instead of an automatically loaded configuration. Configuration can
#                 be automatically loaded when:
#
#                     - The model is a model provided by the library (loaded with the *model id* string of a pretrained
#                       model).
#                     - The model was saved using [`~PreTrainedModel.save_pretrained`] and is reloaded by supplying the
#                       save directory.
#                     - The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
#                       configuration JSON file named *config.json* is found in the directory.
#             cache_dir (`Union[str, os.PathLike]`, *optional*):
#                 Path to a directory in which a downloaded pretrained model configuration should be cached if the
#                 standard cache should not be used.
#             from_pt (`bool`, *optional*, defaults to `False`):
#                 Load the model weights from a PyTorch checkpoint save file (see docstring of
#                 `pretrained_model_name_or_path` argument).
#             ignore_mismatched_sizes (`bool`, *optional*, defaults to `False`):
#                 Whether or not to raise an error if some of the weights from the checkpoint do not have the same size
#                 as the weights of the model (if for instance, you are instantiating a model with 10 labels from a
#                 checkpoint with 3 labels).
#             force_download (`bool`, *optional*, defaults to `False`):
#                 Whether or not to force the (re-)download of the model weights and configuration files, overriding the
#                 cached versions if they exist.
#             resume_download (`bool`, *optional*, defaults to `False`):
#                 Whether or not to delete incompletely received files. Will attempt to resume the download if such a
#                 file exists.
#             proxies (`Dict[str, str]`, *optional*):
#                 A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
#                 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
#             local_files_only(`bool`, *optional*, defaults to `False`):
#                 Whether or not to only look at local files (i.e., do not try to download the model).
#             use_auth_token (`str` or `bool`, *optional*):
#                 The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
#                 the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).
#             revision (`str`, *optional*, defaults to `"main"`):
#                 The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
#                 git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
#                 identifier allowed by git.
#
#
#                 <Tip>
#
#                 To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>".
#
#                 </Tip>
#
#             subfolder (`str`, *optional*, defaults to `""`):
#                 In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
#                 specify the folder name here.
#             kwargs (remaining dictionary of keyword arguments, *optional*):
#                 Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
#                 `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
#                 automatically loaded:
#
#                     - If a configuration is provided with `config`, `**kwargs` will be directly passed to the
#                       underlying model's `__init__` method (we assume all relevant updates to the configuration have
#                       already been done)
#                     - If a configuration is not provided, `kwargs` will be first passed to the configuration class
#                       initialization function ([`~PretrainedConfig.from_pretrained`]). Each key of `kwargs` that
#                       corresponds to a configuration attribute will be used to override said attribute with the
#                       supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
#                       will be passed to the underlying model's `__init__` function.
#
#         Examples:
#
#         ```python
#         >>> from transformers import BertConfig, FlaxBertModel
#
#         >>> # Download model and configuration from huggingface.co and cache.
#         >>> model = FlaxBertModel.from_pretrained("bert-base-cased")
#         >>> # Model was saved using *save_pretrained('./test/saved_model/')* (for example purposes, not runnable).
#         >>> model = FlaxBertModel.from_pretrained("./test/saved_model/")
#         >>> # Loading from a PyTorch checkpoint file instead of a PyTorch model (slower, for example purposes, not runnable).
#         >>> config = BertConfig.from_json_file("./pt_model/config.json")
#         >>> model = FlaxBertModel.from_pretrained("./pt_model/pytorch_model.bin", from_pt=True, config=config)
#         ```"""
#         config = kwargs.pop("config", None)
#         cache_dir = kwargs.pop("cache_dir", None)
#         from_pt = kwargs.pop("from_pt", False)
#         ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
#         force_download = kwargs.pop("force_download", False)
#         resume_download = kwargs.pop("resume_download", False)
#         proxies = kwargs.pop("proxies", None)
#         local_files_only = kwargs.pop("local_files_only", False)
#         use_auth_token = kwargs.pop("use_auth_token", None)
#         revision = kwargs.pop("revision", None)
#         trust_remote_code = kwargs.pop("trust_remote_code", None)
#         from_pipeline = kwargs.pop("_from_pipeline", None)
#         from_auto_class = kwargs.pop("_from_auto", False)
#         _do_init = kwargs.pop("_do_init", True)
#         subfolder = kwargs.pop("subfolder", "")
#         commit_hash = kwargs.pop("_commit_hash", None)
#
#         if trust_remote_code is True:
#             logger.warning(
#                 "The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is"
#                 " ignored."
#             )
#
#         user_agent = {"file_type": "model", "framework": "flax", "from_auto_class": from_auto_class}
#         if from_pipeline is not None:
#             user_agent["using_pipeline"] = from_pipeline
#
#         if is_offline_mode() and not local_files_only:
#             logger.info("Offline mode: forcing local_files_only=True")
#             local_files_only = True
#
#         # Load config if we don't provide a configuration
#         if not isinstance(config, PretrainedConfig):
#             config_path = config if config is not None else pretrained_model_name_or_path
#             config, model_kwargs = cls.config_class.from_pretrained(
#                 config_path,
#                 cache_dir=cache_dir,
#                 return_unused_kwargs=True,
#                 force_download=force_download,
#                 resume_download=resume_download,
#                 proxies=proxies,
#                 local_files_only=local_files_only,
#                 use_auth_token=use_auth_token,
#                 revision=revision,
#                 subfolder=subfolder,
#                 _from_auto=from_auto_class,
#                 _from_pipeline=from_pipeline,
#                 _commit_hash=commit_hash,
#                 **kwargs,
#             )
#         else:
#             model_kwargs = kwargs.copy()
#
#         if commit_hash is None:
#             commit_hash = getattr(config, "_commit_hash", None)
#
#         # Add the dtype to model_kwargs
#         model_kwargs["dtype"] = dtype
#
#         # This variable will flag if we're loading a sharded checkpoint. In this case the archive file is just the
#         # index of the files.
#         is_sharded = False
#
#         # Load model
#         if pretrained_model_name_or_path is not None:
#             pretrained_model_name_or_path = str(pretrained_model_name_or_path)
#             is_local = os.path.isdir(pretrained_model_name_or_path)
#             if os.path.isdir(pretrained_model_name_or_path):
#                 if from_pt and os.path.isfile(os.path.join(pretrained_model_name_or_path, subfolder, WEIGHTS_NAME)):
#                     # Load from a PyTorch checkpoint
#                     archive_file = os.path.join(pretrained_model_name_or_path, subfolder, WEIGHTS_NAME)
#                 elif from_pt and os.path.isfile(
#                     os.path.join(pretrained_model_name_or_path, subfolder, WEIGHTS_INDEX_NAME)
#                 ):
#                     # Load from a sharded pytorch checkpoint
#                     archive_file = os.path.join(pretrained_model_name_or_path, subfolder, WEIGHTS_INDEX_NAME)
#                     is_sharded = True
#                 elif os.path.isfile(os.path.join(pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_NAME)):
#                     # Load from a Flax checkpoint
#                     archive_file = os.path.join(pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_NAME)
#                 elif os.path.isfile(os.path.join(pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_INDEX_NAME)):
#                     # Load from a sharded Flax checkpoint
#                     archive_file = os.path.join(pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_INDEX_NAME)
#                     is_sharded = True
#                 # At this stage we don't have a weight file so we will raise an error.
#                 elif os.path.isfile(os.path.join(pretrained_model_name_or_path, subfolder, WEIGHTS_NAME)):
#                     raise EnvironmentError(
#                         f"Error no file named {FLAX_WEIGHTS_NAME} found in directory {pretrained_model_name_or_path} "
#                         "but there is a file for PyTorch weights. Use `from_pt=True` to load this model from those "
#                         "weights."
#                     )
#                 else:
#                     raise EnvironmentError(
#                         f"Error no file named {FLAX_WEIGHTS_NAME} or {WEIGHTS_NAME} found in directory "
#                         f"{pretrained_model_name_or_path}."
#                     )
#             elif os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path)):
#                 archive_file = pretrained_model_name_or_path
#                 is_local = True
#             elif is_remote_url(pretrained_model_name_or_path):
#                 filename = pretrained_model_name_or_path
#                 resolved_archive_file = download_url(pretrained_model_name_or_path)
#             else:
#                 filename = WEIGHTS_NAME if from_pt else FLAX_WEIGHTS_NAME
#                 try:
#                     # Load from URL or cache if already cached
#                     cached_file_kwargs = {
#                         "cache_dir": cache_dir,
#                         "force_download": force_download,
#                         "proxies": proxies,
#                         "resume_download": resume_download,
#                         "local_files_only": local_files_only,
#                         "use_auth_token": use_auth_token,
#                         "user_agent": user_agent,
#                         "revision": revision,
#                         "subfolder": subfolder,
#                         "_raise_exceptions_for_missing_entries": False,
#                         "_commit_hash": commit_hash,
#                     }
#                     resolved_archive_file = cached_file(pretrained_model_name_or_path, filename, **cached_file_kwargs)
#
#                     # Since we set _raise_exceptions_for_missing_entries=False, we don't get an expection but a None
#                     # result when internet is up, the repo and revision exist, but the file does not.
#                     if resolved_archive_file is None and filename == FLAX_WEIGHTS_NAME:
#                         # Maybe the checkpoint is sharded, we try to grab the index name in this case.
#                         resolved_archive_file = cached_file(
#                             pretrained_model_name_or_path, FLAX_WEIGHTS_INDEX_NAME, **cached_file_kwargs
#                         )
#                         if resolved_archive_file is not None:
#                             is_sharded = True
#                     # Maybe the checkpoint is pytorch sharded, we try to grab the pytorch index name in this case.
#                     elif resolved_archive_file is None and from_pt:
#                         resolved_archive_file = cached_file(
#                             pretrained_model_name_or_path, WEIGHTS_INDEX_NAME, **cached_file_kwargs
#                         )
#                         if resolved_archive_file is not None:
#                             is_sharded = True
#                     if resolved_archive_file is None:
#                         # Otherwise, maybe there is a TF or Flax model file.  We try those to give a helpful error
#                         # message.
#                         has_file_kwargs = {
#                             "revision": revision,
#                             "proxies": proxies,
#                             "use_auth_token": use_auth_token,
#                         }
#                         if has_file(pretrained_model_name_or_path, WEIGHTS_NAME, **has_file_kwargs):
#                             raise EnvironmentError(
#                                 f"{pretrained_model_name_or_path} does not appear to have a file named"
#                                 f" {FLAX_WEIGHTS_NAME} but there is a file for PyTorch weights. Use `from_pt=True` to"
#                                 " load this model from those weights."
#                             )
#                         elif has_file(pretrained_model_name_or_path, WEIGHTS_INDEX_NAME, **has_file_kwargs):
#                             raise EnvironmentError(
#                                 f"{pretrained_model_name_or_path} does not appear to have a file named"
#                                 f" {FLAX_WEIGHTS_INDEX_NAME} but there is a sharded file for PyTorch weights. Use"
#                                 " `from_pt=True` to load this model from those weights."
#                             )
#                         else:
#                             raise EnvironmentError(
#                                 f"{pretrained_model_name_or_path} does not appear to have a file named"
#                                 f" {FLAX_WEIGHTS_NAME} or {WEIGHTS_NAME}."
#                             )
#                 except EnvironmentError:
#                     # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted
#                     # to the original exception.
#                     raise
#                 except Exception:
#                     # For any other exception, we throw a generic error.
#                     raise EnvironmentError(
#                         f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it"
#                         " from 'https://huggingface.co/models', make sure you don't have a local directory with the"
#                         f" same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a"
#                         f" directory containing a file named {FLAX_WEIGHTS_NAME} or {WEIGHTS_NAME}."
#                     )
#
#             if is_local:
#                 logger.info(f"loading weights file {archive_file}")
#                 resolved_archive_file = archive_file
#             else:
#                 logger.info(f"loading weights file {filename} from cache at {resolved_archive_file}")
#         else:
#             resolved_archive_file = None
#
#         # We'll need to download and cache each checkpoint shard if the checkpoint is sharded.
#         if is_sharded:
#             # resolved_archive_file becomes a list of files that point to the different checkpoint shards in this case.
#             resolved_archive_file, _ = get_checkpoint_shard_files(
#                 pretrained_model_name_or_path,
#                 resolved_archive_file,
#                 cache_dir=cache_dir,
#                 force_download=force_download,
#                 proxies=proxies,
#                 resume_download=resume_download,
#                 local_files_only=local_files_only,
#                 use_auth_token=use_auth_token,
#                 user_agent=user_agent,
#                 revision=revision,
#                 subfolder=subfolder,
#                 _commit_hash=commit_hash,
#             )
#
#         # init random models
#         model = cls(config, *model_args, _do_init=_do_init, **model_kwargs)
#
#         if from_pt:
#             state = load_pytorch_checkpoint_in_flax_state_dict(model, resolved_archive_file, is_sharded)
#         else:
#             if is_sharded:
#                 state = cls.load_flax_sharded_weights(resolved_archive_file)
#             else:
#                 try:
#                     with open(resolved_archive_file, "rb") as state_f:
#                         state = from_bytes(cls, state_f.read())
#                 except (UnpicklingError, msgpack.exceptions.ExtraData) as e:
#                     try:
#                         with open(resolved_archive_file) as f:
#                             if f.read().startswith("version"):
#                                 raise OSError(
#                                     "You seem to have cloned a repository without having git-lfs installed. Please"
#                                     " install git-lfs and run `git lfs install` followed by `git lfs pull` in the"
#                                     " folder you cloned."
#                                 )
#                             else:
#                                 raise ValueError from e
#                     except (UnicodeDecodeError, ValueError):
#                         raise EnvironmentError(f"Unable to convert {archive_file} to Flax deserializable object. ")
#             # make sure all arrays are stored as jnp.arrays
#             # NOTE: This is to prevent a bug this will be fixed in Flax >= v0.3.4:
#             # https://github.com/google/flax/issues/1261
#             if _do_init:
#                 state = jax.tree_util.tree_map(jnp.array, state)
#             else:
#                 # keep the params on CPU if we don't want to initialize
#                 state = jax.tree_util.tree_map(lambda x: jax.device_put(x, jax.devices("cpu")[0]), state)
#
#         if ("batch_stats" in state) or ("spectral_stats" in state):  # if flax model contains batch norm layers or spectral norm layer
#             stats_name = "batch_stats" if "batch_stats" in state else "spectral_stats"
#             # if model is base model only use model_prefix key
#             if (
#                 cls.base_model_prefix not in dict(model.params_shape_tree["params"])
#                 and cls.base_model_prefix in state["params"]
#             ):
#                 state["params"] = state["params"][cls.base_model_prefix]
#                 state["batch_stats"] = state["batch_stats"][cls.base_model_prefix]
#
#             # if model is head model and we are loading weights from base model
#             # we initialize new params dict with base_model_prefix
#             if (
#                 cls.base_model_prefix in dict(model.params_shape_tree["params"])
#                 and cls.base_model_prefix not in state["params"]
#             ):
#                 state = {
#                     "params": {cls.base_model_prefix: state["params"]},
#                     "batch_stats": {cls.base_model_prefix: state["batch_stats"]},
#                 }
#         else:
#             # if model is base model only use model_prefix key
#             if cls.base_model_prefix not in dict(model.params_shape_tree) and cls.base_model_prefix in state:
#                 state = state[cls.base_model_prefix]
#
#             # if model is head model and we are loading weights from base model
#             # we initialize new params dict with base_model_prefix
#             if cls.base_model_prefix in dict(model.params_shape_tree) and cls.base_model_prefix not in state:
#                 state = {cls.base_model_prefix: state}
#
#         # flatten dicts
#         state = flatten_dict(state)
#
#         random_state = flatten_dict(unfreeze(model.params if _do_init else model.params_shape_tree))
#
#         missing_keys = model.required_params - set(state.keys())
#         unexpected_keys = set(state.keys()) - model.required_params
#
#         # Disabling warning when porting pytorch weights to flax, flax does not uses num_batches_tracked
#         for unexpected_key in unexpected_keys.copy():
#             if "num_batches_tracked" in unexpected_key[-1]:
#                 unexpected_keys.remove(unexpected_key)
#
#         if missing_keys and not _do_init:
#             logger.warning(
#                 f"The checkpoint {pretrained_model_name_or_path} is missing required keys: {missing_keys}. "
#                 "Make sure to call model.init_weights to initialize the missing weights."
#             )
#             cls._missing_keys = missing_keys
#
#         # Mistmatched keys contains tuples key/shape1/shape2 of weights in the checkpoint that have a shape not
#         # matching the weights in the model.
#         mismatched_keys = []
#         for key in state.keys():
#             if key in random_state and state[key].shape != random_state[key].shape:
#                 if ignore_mismatched_sizes:
#                     mismatched_keys.append((key, state[key].shape, random_state[key].shape))
#                     state[key] = random_state[key]
#                 else:
#                     raise ValueError(
#                         f"Trying to load the pretrained weight for {key} failed: checkpoint has shape "
#                         f"{state[key].shape} which is incompatible with the model shape {random_state[key].shape}. "
#                         "Using `ignore_mismatched_sizes=True` if you really want to load this checkpoint inside this "
#                         "model."
#                     )
#
#         # add missing keys as random parameters if we are initializing
#         if missing_keys and _do_init:
#             for missing_key in missing_keys:
#                 state[missing_key] = random_state[missing_key]
#
#         # remove unexpected keys to not be saved again
#         for unexpected_key in unexpected_keys:
#             del state[unexpected_key]
#
#         if len(unexpected_keys) > 0:
#             logger.warning(
#                 f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when"
#                 f" initializing {model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are"
#                 f" initializing {model.__class__.__name__} from the checkpoint of a model trained on another task or"
#                 " with another architecture (e.g. initializing a BertForSequenceClassification model from a"
#                 " BertForPreTraining model).\n- This IS NOT expected if you are initializing"
#                 f" {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly identical"
#                 " (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
#             )
#         else:
#             logger.info(f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n")
#
#         if len(missing_keys) > 0:
#             logger.warning(
#                 f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
#                 f" {pretrained_model_name_or_path} and are newly initialized: {missing_keys}\nYou should probably"
#                 " TRAIN this model on a down-stream task to be able to use it for predictions and inference."
#             )
#         elif len(mismatched_keys) == 0:
#             logger.info(
#                 f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at"
#                 f" {pretrained_model_name_or_path}.\nIf your task is similar to the task the model of the checkpoint"
#                 f" was trained on, you can already use {model.__class__.__name__} for predictions without further"
#                 " training."
#             )
#         if len(mismatched_keys) > 0:
#             mismatched_warning = "\n".join(
#                 [
#                     f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
#                     for key, shape1, shape2 in mismatched_keys
#                 ]
#             )
#             logger.warning(
#                 f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
#                 f" {pretrained_model_name_or_path} and are newly initialized because the shapes did not"
#                 f" match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be able"
#                 " to use it for predictions and inference."
#             )
#
#         # dictionary of key: dtypes for the model params
#         param_dtypes = jax.tree_util.tree_map(lambda x: x.dtype, state)
#         # extract keys of parameters not in jnp.float32
#         fp16_params = [k for k in param_dtypes if param_dtypes[k] == jnp.float16]
#         bf16_params = [k for k in param_dtypes if param_dtypes[k] == jnp.bfloat16]
#
#         # raise a warning if any of the parameters are not in jnp.float32
#         if len(fp16_params) > 0:
#             logger.warning(
#                 f"Some of the weights of {model.__class__.__name__} were initialized in float16 precision from "
#                 f"the model checkpoint at {pretrained_model_name_or_path}:\n{fp16_params}\n"
#                 "You should probably UPCAST the model weights to float32 if this was not intended. "
#                 "See [`~FlaxPreTrainedModel.to_fp32`] for further information on how to do this."
#             )
#
#         if len(bf16_params) > 0:
#             logger.warning(
#                 f"Some of the weights of {model.__class__.__name__} were initialized in bfloat16 precision from "
#                 f"the model checkpoint at {pretrained_model_name_or_path}:\n{bf16_params}\n"
#                 "You should probably UPCAST the model weights to float32 if this was not intended. "
#                 "See [`~FlaxPreTrainedModel.to_fp32`] for further information on how to do this."
#             )
#
#         # If it is a model with generation capabilities, attempt to load the generation config
#         if model.can_generate():
#             try:
#                 model.generation_config = GenerationConfig.from_pretrained(
#                     pretrained_model_name_or_path,
#                     cache_dir=cache_dir,
#                     force_download=force_download,
#                     resume_download=resume_download,
#                     proxies=proxies,
#                     local_files_only=local_files_only,
#                     use_auth_token=use_auth_token,
#                     revision=revision,
#                     subfolder=subfolder,
#                     _from_auto=from_auto_class,
#                     _from_pipeline=from_pipeline,
#                     **kwargs,
#                 )
#             except OSError:
#                 logger.info(
#                     "Generation config file not found, using a generation config created from the model config."
#                 )
#                 pass
#
#         if _do_init:
#             # set correct parameters
#             model.params = unflatten_dict(state)
#             return model
#         else:
#             return model, unflatten_dict(state)