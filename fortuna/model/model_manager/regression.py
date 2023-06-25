from typing import (
    Dict,
    Optional,
    Tuple,
    Union,
)

from flax.core import FrozenDict
import flax.linen as nn
import jax
from jax import random
from jax._src.prng import PRNGKeyArray
import jax.numpy as jnp
from optax._src.base import PyTree

from fortuna.model.model_manager.base import ModelManager
from fortuna.typing import (
    Array,
    Mutable,
    Params,
)
from fortuna.utils.data import get_inputs_from_shape


class RegressionModelManager(ModelManager):
    def __init__(
        self,
        model: nn.Module,
        likelihood_log_variance_model: nn.Module,
        model_editor: Optional[nn.Module] = None,
    ):
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
        super(RegressionModelManager, self).__init__(model, model_editor)
        self.likelihood_log_variance_model = likelihood_log_variance_model

    def apply(
        self,
        params: Params,
        inputs: Array,
        mutable: Optional[Mutable] = None,
        train: bool = False,
        rng: Optional[PRNGKeyArray] = None,
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, PyTree]]:
        # setup dropout key
        if rng is not None:
            rng, model_dropout_key, lik_log_var_dropout_key = random.split(rng, 3)
            model_rngs = {"dropout": model_dropout_key}
            lik_log_var_rngs = {"dropout": lik_log_var_dropout_key}
        else:
            model_rngs = None
            lik_log_var_rngs = None

        if mutable is not None:
            mutable = mutable.unfreeze()
            mutable["model"] = mutable.get("model")
            mutable["lik_log_var"] = mutable.get("lik_log_var")

        model_has_aux = train and mutable is not None and mutable["model"] is not None
        lik_log_var_has_aux = (
            train and mutable is not None and mutable["lik_log_var"] is not None
        )

        def apply_fn(p, x, m_mutable, llv_mutable):
            model_outputs = self.model.apply(
                p["model"],
                x,
                train=train,
                mutable=m_mutable,
                rngs=model_rngs,
            )
            lik_log_var_outputs = self.likelihood_log_variance_model.apply(
                p["lik_log_var"],
                x,
                train=train,
                mutable=llv_mutable,
                rngs=lik_log_var_rngs,
            )

            if isinstance(model_outputs, tuple) and not model_has_aux:
                model_outputs = model_outputs[0]
            if isinstance(lik_log_var_outputs, tuple) and not lik_log_var_has_aux:
                lik_log_var_outputs = lik_log_var_outputs[0]

            if model_has_aux:
                model_outputs, m_mutable = model_outputs
            if lik_log_var_has_aux:
                lik_log_var_outputs, llv_mutable = lik_log_var_outputs

            self._check_outputs(model_outputs, lik_log_var_outputs)

            aux = dict()
            if train and (m_mutable or llv_mutable):
                aux["mutable"] = dict()
                if m_mutable:
                    aux["mutable"]["model"] = m_mutable
                if llv_mutable:
                    aux["mutable"]["lik_log_var"] = llv_mutable

            return jnp.concatenate(
                (model_outputs, lik_log_var_outputs), axis=-1
            ), FrozenDict(aux)

        if self.model_editor is not None:
            outputs, aux = self.model_editor.apply(
                params["model_editor"],
                apply_fn=lambda p, x: apply_fn(
                    p,
                    x,
                    m_mutable=mutable["model"] if mutable is not None else False,
                    llv_mutable=mutable["lik_log_var"]
                    if mutable is not None
                    else False,
                ),
                model_params=params,
                x=inputs,
                has_aux=True,
            )
        else:
            outputs, aux = apply_fn(
                params,
                inputs,
                m_mutable=mutable["model"] if mutable is not None else False,
                llv_mutable=mutable["lik_log_var"] if mutable is not None else False,
            )

        if len(aux) > 0:
            return outputs, aux
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
        params = dict(
            model=self.model.init(model_rngs, jnp.zeros((1,) + input_shape), **kwargs),
            lik_log_var=self.likelihood_log_variance_model.init(
                lik_log_var_rngs, jnp.zeros((1,) + input_shape), **kwargs
            ),
        )

        def apply_fn(p, x):
            model_outputs = self.model.apply(
                p["model"],
                x,
                rngs=model_rngs,
            )
            lik_log_var_outputs = self.likelihood_log_variance_model.apply(
                p["lik_log_var"],
                x,
                rngs=lik_log_var_rngs,
            )

            self._check_outputs(model_outputs, lik_log_var_outputs)

            return jnp.concatenate((model_outputs, lik_log_var_outputs), axis=-1)

        if self.model_editor is not None:
            if rng is None:
                rng = self.rng
            rng, params_key, dropout_key = random.split(rng, 3)
            rngs = {"params": params_key, "dropout": dropout_key}
            params.update(
                dict(
                    model_editor=self.model_editor.init(
                        rngs,
                        apply_fn=apply_fn,
                        model_params=params,
                        x=get_inputs_from_shape(input_shape),
                        has_aux=False,
                    )
                )
            )
        return params

    def _check_outputs(
        self, model_outputs: jnp.ndarray, lik_log_var_outputs: jnp.ndarray
    ) -> None:
        if model_outputs.shape[-1] != lik_log_var_outputs.shape[-1]:
            raise ValueError(
                f"""The output dimensions of `model` and `likelihood_log_variance_model must be the same.
            However, {model_outputs.shape[-1]} and {lik_log_var_outputs.shape[-1]} were found, respectively."""
            )
