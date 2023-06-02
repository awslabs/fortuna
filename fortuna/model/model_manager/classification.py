from functools import partial
import logging
from typing import (
    Dict,
    Mapping,
    Optional,
    Tuple,
    Union,
)

from flax.core import FrozenDict
import flax.linen as nn
from flax.training.checkpoints import PyTree
import jax
from jax import random
from jax._src.prng import PRNGKeyArray
import jax.numpy as jnp

from fortuna.model.model_manager.base import ModelManager
from fortuna.model.utils.random_features import RandomFeatureGaussianProcess
from fortuna.typing import (
    Array,
    Mutable,
    Params,
)
from fortuna.utils.data import get_inputs_from_shape
from fortuna.utils.nested_dicts import nested_update

logger = logging.getLogger(__name__)


class ClassificationModelManager(ModelManager):
    def __init__(self, model: nn.Module, model_editor: Optional[nn.Module] = None):
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
        super(ClassificationModelManager, self).__init__(model, model_editor)

    def apply(
        self,
        params: Params,
        inputs: Array,
        mutable: Optional[Mutable] = None,
        train: bool = False,
        rng: Optional[PRNGKeyArray] = None,
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, PyTree]]:
        variables = params["model"].unfreeze()

        # setup dropout key
        if rng is not None:
            rng, dropout_rng = random.split(rng, 2)
            rngs = {"dropout": dropout_rng}
        else:
            rngs = None

        if mutable is not None:
            mutable_variables = mutable["model"].unfreeze()
            variables.update(mutable_variables)
            mutable = list(mutable_variables.keys())

        def apply_fn(v, x):
            _outputs = self.model.apply(
                v,
                x,
                mutable=mutable if mutable is not None else False,
                train=train,
                rngs=rngs,
            )
            if isinstance(_outputs, tuple) and not has_aux:
                _outputs = _outputs[0]
            return _outputs

        has_aux = train and mutable is not None

        if self.model_editor is not None:
            outputs = self.model_editor.apply(
                params["model_editor"],
                apply_fn=apply_fn,
                model_params=variables,
                x=inputs,
                has_aux=has_aux,
            )
        else:
            outputs = apply_fn(variables, inputs)

        if has_aux:
            outputs, mutable = outputs
            return outputs, {"mutable": FrozenDict({"model": mutable})}
        return outputs

    def init(
        self, input_shape: Tuple[int, ...], rng: Optional[PRNGKeyArray] = None, **kwargs
    ) -> Dict[str, Mapping]:
        if rng is None:
            rng = self.rng.get()
        rng, params_key, dropout_key = random.split(rng, 3)
        rngs = {"params": params_key, "dropout": dropout_key}
        params = dict(
            model=self.model.init(rngs, jnp.zeros((1,) + input_shape), **kwargs)
        )
        if self.model_editor is not None:
            if rng is None:
                rng = self.rng
            rng, params_key, dropout_key = random.split(rng, 3)
            rngs = {"params": params_key, "dropout": dropout_key}
            params.update(
                dict(
                    model_editor=self.model_editor.init(
                        rngs,
                        apply_fn=self.model.apply,
                        model_params=params["model"],
                        x=get_inputs_from_shape(input_shape),
                        has_aux=False,
                    )
                )
            )
        return params


class SNGPClassificationModelManagerMixin:
    def __init__(
        self,
        *args,
        output_dim: int,
        gp_hidden_features: int = 1024,
        normalize_input: bool = False,
        ridge_penalty: float = 1.0,
        momentum: Optional[float] = None,
        mean_field_factor: float = 1.0,
        **kwargs,
    ):
        """
        Classification model manager Mixin for SNGP models.

        Parameters
        ----------
        output_dim: int
            The output dimension of the network.
        normalize_input: bool
            Whether to normalize the input using nn.LayerNorm.
        gp_hidden_features: int
            The number of random fourier features.
        ridge_penalty: float
            Initial Ridge penalty to weight covariance matrix.
            This value is used to stabilize the eigenvalues of weight covariance estimate :math:`\Sigma` so that
            the matrix inverse can be computed for :math:`\Sigma = (\mathbf{I}*s+\mathbf{X}^T\mathbf{X})^{-1}`.
            The ridge factor :math:`s` cannot be too large since otherwise it will dominate
            making the covariance estimate not meaningful.
        momentum: Optional[float]
            A discount factor used to compute the moving average for posterior
            precision matrix. Analogous to the momentum factor in batch normalization.
            If `None` then update covariance matrix using a naive sum without
            momentum, which is desirable if the goal is to compute the exact
            covariance matrix by passing through data once (say in the final epoch).
            In this case, make sure to reset the precision matrix variable between
            epochs to avoid double counting.
        mean_field_factor: float
            The scale factor for mean-field approximation, used to adjust (at inference time) the influence of
            posterior variance in posterior mean approximation.
            See `Zhiyun L. et al., 2020 <https://arxiv.org/abs/2006.07584>`_ for more details.
        """
        super(SNGPClassificationModelManagerMixin, self).__init__(*args, **kwargs)
        self.output_dim = output_dim
        self.gp_hidden_features = gp_hidden_features
        self.normalize_input = normalize_input
        self.ridge_penalty = ridge_penalty
        self.momentum = momentum
        self.mean_field_factor = mean_field_factor
        self._gp_output_model = self._get_output_model()
        self._gp_output_model_mutable_keys = [
            "sngp_random_features",
            "sngp_laplace_covariance",
        ]

    def _get_output_model(self) -> RandomFeatureGaussianProcess:
        return RandomFeatureGaussianProcess(
            features=self.output_dim,
            hidden_features=self.gp_hidden_features,
            normalize_input=self.normalize_input,
            hidden_kwargs={"collection_name": "sngp_random_features"},
            covariance_kwargs={
                "ridge_penalty": self.ridge_penalty,
                "momentum": self.momentum,
                "collection_name": "sngp_laplace_covariance",
            },
        )

    def _mean_field_logits(
        self, logits: Array, covariance: Optional[Array] = None
    ) -> Array:
        """
        Adjust the model logits s.t. its softmax approximates the posterior mean
        (`Zhiyun L. et al., 2020 <https://arxiv.org/abs/2006.07584>`_).

        Parameters
        ----------
          logits: Array
            A float tensor of shape (batch_size, num_classes).
          covariance: Array
            A float tensor of shape (batch_size, batch_size) or (batch_size,). If None then it
            assumes the covariance is an identity matrix.

        Returns
        ----------
        Array
            The adjusted model logits.
        """
        # Implementation adapted from https://github.com/google/edward2/blob/520e28285e905e0021e49b52b982ee5ea170071c/edward2/tensorflow/layers/utils.py#L379
        if self.mean_field_factor < 0:
            raise ValueError(f"mean_field_factor cannot be < 0.")

        # Compute standard deviation.
        if covariance is None:
            variances = 1.0
        else:
            variances = jnp.diagonal(covariance) if covariance.ndim == 2 else covariance

        # Compute scaling coefficient for mean-field approximation.
        logits_scale = jnp.sqrt(1.0 + variances * self.mean_field_factor)

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
        if mutable and mutable is not None:
            deep_feature_extractor_mutable = {
                "model": FrozenDict(
                    {
                        k: v
                        for k, v in mutable["model"].items()
                        if k not in self._gp_output_model_mutable_keys
                    }
                )
            }
            gp_model_mutable = {
                k: v
                for k, v in mutable["model"].items()
                if k in self._gp_output_model_mutable_keys
            }
        else:
            deep_feature_extractor_mutable = mutable

        has_aux = mutable is not None and train

        def apply_fn(p, x):
            return super(SNGPClassificationModelManagerMixin, self).apply(
                p, x, deep_feature_extractor_mutable, train, rng
            )

        if self.model_editor is not None:
            deep_feature_extractor_outputs = self.model_editor.apply(
                params["model_editor"],
                apply_fn=apply_fn,
                model_params=params,
                x=inputs,
                has_aux=has_aux,
            )
        else:
            deep_feature_extractor_outputs = apply_fn(params, inputs)

        variables = params["model"].unfreeze()
        if mutable is not None:
            variables.update(gp_model_mutable)
            mutable = list(gp_model_mutable.keys())

        if has_aux:
            (
                deep_feature_extractor_outputs,
                deep_feature_extractor_mutable,
            ) = deep_feature_extractor_outputs

        if has_aux:
            outputs, gp_mutable = self._gp_output_model.apply(
                variables, deep_feature_extractor_outputs, mutable=mutable
            )
            outputs = outputs[0]
            if gp_mutable is not None:
                mutable = deep_feature_extractor_mutable["mutable"]["model"].unfreeze()
                mutable.update(gp_mutable.unfreeze())
                mutable = {"mutable": FrozenDict({"model": mutable})}
            return outputs, mutable
        else:
            logits, covariance = self._gp_output_model.apply(
                variables,
                deep_feature_extractor_outputs,
                mutable=False,
                return_full_covariance=False,
            )
            return self._mean_field_logits(logits, covariance)


class SNGPClassificationModelManager(
    SNGPClassificationModelManagerMixin, ClassificationModelManager
):
    def __init__(self, model: nn.Module, *args, **kwargs):
        """
        Classification model manager for SNGP models.

        Parameters
        ----------
        model : nn.Module
            A model describing the deterministic relation between inputs and outputs. The outputs of the model
            is the latent representation of the input, which in this case, does not correspond to the logits of a
            softmax probability vector. The output dimension of the model is not dependent on the number
            of classes in the classification task.
        """
        super(SNGPClassificationModelManager, self).__init__(model, *args, **kwargs)

    def init(
        self, input_shape: Tuple[int, ...], rng: Optional[PRNGKeyArray] = None, **kwargs
    ) -> Dict[str, FrozenDict]:
        if rng is None:
            rng = self.rng.get()
        rng, params_key, dropout_key = random.split(rng, 3)
        rngs = {"params": params_key, "dropout": dropout_key}
        model_params = self.model.init(rngs, jnp.zeros((1,) + input_shape), **kwargs)
        output_shape = jax.eval_shape(
            partial(self.model.apply, train=False),
            model_params,
            jnp.zeros((1,) + input_shape),
        ).shape
        if len(output_shape[1:]) > 1:  # drop batch size
            raise ValueError(
                f"The output shape for the given model is {output_shape}.\n"
                f"In order to use SNGP the output shape of the provide model has to be of shape"
                f"(batch_size, n_features)."
            )
        gp_params = self._gp_output_model.init(rngs, jnp.zeros(output_shape), **kwargs)
        params = dict(
            model=FrozenDict(
                nested_update(model_params.unfreeze(), gp_params.unfreeze())
            )
        )

        if self.model_editor is not None:
            rng, params_key, dropout_key = random.split(rng, 3)
            rngs = {"params": params_key, "dropout": dropout_key}
            params.update(
                dict(
                    model_editor=self.model_editor.init(
                        rngs,
                        apply_fn=self.model.apply,
                        model_params=params["model"],
                        x=get_inputs_from_shape(input_shape),
                        has_aux=False,
                    )
                )
            )
        return params
