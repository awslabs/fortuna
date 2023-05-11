from typing import (
    Dict,
    Optional,
    Tuple,
    Union,
)

from flax.core import FrozenDict
import flax.linen as nn
from flax.training.checkpoints import PyTree
from jax import random
from jax._src.prng import PRNGKeyArray
import jax.numpy as jnp

from fortuna.model.model_manager.base import ModelManager
from fortuna.typing import (
    Array,
    Mutable,
    Params,
)


class RegressionModelManager(ModelManager):
    def __init__(self, model: nn.Module, likelihood_log_variance_model: nn.Module):
        r"""
        Regression model manager class. It orchestrates the forward pass of the model in the probabilistic model.

        Parameters
        ----------
        model : nn.Module
            A model describing the deterministic relation between inputs and outputs. It characterizes the mean model
            of the likelihood function. The outputs must belong to the same space as the target variables.
            Let :math:`x` be input variables and :math:`w` the random model parameters. Then the model is described by
            a function :math:`\mu(w, x)`.
        likelihood_log_variance_model: nn.Module
            A model characterizing the log-variance of a Gaussian likelihood function. The outputs must belong to the
            same space as the target variables. Let :math:`x` be input variables and :math:`w` the random model
            parameters. Then the model is described by a function :math:`\log\sigma^2(w, x)`.
        """
        self.model = model
        self.likelihood_log_variance_model = likelihood_log_variance_model

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
        model_variables = params["model"].unfreeze()
        lik_log_var_variables = params["lik_log_var"].unfreeze()

        # setup dropout key
        if rng is not None:
            rng, model_dropout_key, lik_log_var_dropout_key = random.split(rng, 3)
            model_rngs = {"dropout": model_dropout_key}
            lik_log_var_rngs = {"dropout": lik_log_var_dropout_key}
        else:
            model_rngs = None
            lik_log_var_rngs = None

        if mutable:
            try:
                model_mutable_variables = mutable["model"].unfreeze()
                model_variables.update(model_mutable_variables)
                model_mutable = list(model_mutable_variables.keys())
            except KeyError:
                model_mutable = False
            try:
                lik_log_var_mutable_variables = mutable["lik_log_var"].unfreeze()
                lik_log_var_variables.update(lik_log_var_mutable_variables)
                lik_log_var_mutable = list(lik_log_var_mutable_variables.keys())
            except KeyError:
                lik_log_var_mutable = False
        else:
            model_mutable = False
            lik_log_var_mutable = False

        aux = {}
        if train and model_mutable:
            model_outputs, model_mutable = self.model.apply(
                model_variables,
                inputs,
                train=train,
                mutable=model_mutable,
                rngs=model_rngs,
            )
            aux.update({"model": model_mutable})
        else:
            model_outputs = self.model.apply(
                model_variables, inputs, train=train, mutable=False, rngs=model_rngs
            )
        if train and lik_log_var_mutable:
            (
                lik_log_var_outputs,
                lik_log_var_mutable,
            ) = self.likelihood_log_variance_model.apply(
                lik_log_var_variables,
                train=train,
                mutable=lik_log_var_mutable,
                rngs=lik_log_var_rngs,
            )
            aux.update({"likelihood_log_variance_model": lik_log_var_mutable})
        else:
            lik_log_var_outputs = self.likelihood_log_variance_model.apply(
                lik_log_var_variables,
                inputs,
                train=train,
                mutable=False,
                rngs=lik_log_var_rngs,
            )

        outputs = jnp.concatenate((model_outputs, lik_log_var_outputs), axis=-1)
        if model_outputs.shape[-1] != lik_log_var_outputs.shape[-1]:
            raise ValueError(
                f"""The output dimensions of `model` and `likelihood_log_variance_model must be the same.
            However, {model_outputs.shape[-1]} and {lik_log_var_outputs.shape[-1]} were found, respectively."""
            )
        if len(aux) > 0:
            return outputs, {"mutable": FrozenDict(aux)}
        return outputs

    def init(
        self, input_shape: Tuple, rng: Optional[PRNGKeyArray] = None, **kwargs
    ) -> Dict[str, FrozenDict]:
        if rng is None:
            rng = self.rng.get()
        (
            rng,
            model_params_key,
            model_dropout_key,
            lik_log_var_params_key,
            lik_log_var_dropout_key,
        ) = random.split(rng, 5)
        model_rngs = {"params": model_params_key, "dropout": model_dropout_key}
        lik_log_var_rngs = {
            "params": lik_log_var_params_key,
            "dropout": lik_log_var_params_key,
        }
        return dict(
            model=self.model.init(model_rngs, jnp.zeros((1,) + input_shape), **kwargs),
            lik_log_var=self.likelihood_log_variance_model.init(
                lik_log_var_rngs, jnp.zeros((1,) + input_shape), **kwargs
            ),
        )
