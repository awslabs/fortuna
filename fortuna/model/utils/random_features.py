# implementation adapted from https://github.com/google/edward2/blob/main/edward2/jax/nn/random_feature.py

import dataclasses
import functools
from typing import Any, Callable, Mapping, Optional, Tuple, Type

import flax.linen as nn
import jax.numpy as jnp
from jax import lax, random
from jax.random import PRNGKeyArray

from fortuna.typing import Array, Shape

linalg = lax.linalg

# Default config for random features.
default_rbf_bias_init = nn.initializers.uniform(scale=2.0 * jnp.pi)
# Using "he_normal" style random feature distribution (see https://arxiv.org/abs/1502.01852).
# Effectively, this is equivalent to approximating a RBF kernel but with the input standardized by
# its dimensionality (i.e., input_scaled = input * sqrt(2. / dim_input)) and
# empirically leads to better performance for neural network inputs.
# default_rbf_kernel_init = nn.initializers.variance_scaling(
#     scale=2.0, mode="fan_in", distribution="normal"
# )
default_rbf_kernel_init = nn.initializers.normal(stddev=1.0)

# Default field value for kwargs, to be used for data class declaration.
default_kwarg_dict = lambda: dataclasses.field(default_factory=dict)

SUPPORTED_LIKELIHOOD = ("binary_logistic", "poisson", "gaussian")


class RandomFeatureGaussianProcess(nn.Module):
    """
    A Gaussian process layer using random Fourier Features.
    
    See `Simple and Principled Uncertainty Estimation with Deterministic 
    Deep Learning via Distance Awareness <https://arxiv.org/abs/2006.10108>`_

    Attributes
    ----------
    features: int
        The number of output units.
    hidden_features: int
        The number of hidden random fourier features.
    normalize_input: bool
        Whether to normalize the input using nn.LayerNorm.
    norm_kwargs: Mapping[str, Any]
        Optional keyword arguments to the input nn.LayerNorm layer.
    hidden_kwargs: Mapping[str, Any]
        Optional keyword arguments to the random feature layer.
    output_kwargs: Mapping[str, Any]
        Optional keyword arguments to the predictive logit layer.
    covariance_kwargs: Mapping[str, Any]
        Optional keyword arguments to the predictive covariance layer.
    """
    features: int
    hidden_features: int = 1024
    normalize_input: bool = False

    # Optional keyword arguments.
    norm_kwargs: Mapping[str, Any] = default_kwarg_dict()
    hidden_kwargs: Mapping[str, Any] = default_kwarg_dict()
    output_kwargs: Mapping[str, Any] = default_kwarg_dict()
    covariance_kwargs: Mapping[str, Any] = default_kwarg_dict()

    def setup(self):
        # pylint:disable=invalid-name,not-a-mapping
        if self.normalize_input:
            # Prefer a parameter-free version of LayerNorm by default
            # (see `Xu et al., 2019 <https://papers.nips.cc/paper/2019/file/2f4fe03d77724a7217006e5d16728874-Paper.pdf>`_)
            # Can be overwritten by passing norm_kwargs=dict(use_bias=..., use_scales=...).
            LayerNorm = functools.partial(nn.LayerNorm, use_bias=False, use_scale=False)
            self.sngp_norm_layer = LayerNorm(**self.norm_kwargs)

        self.sngp_random_features_layer = RandomFourierFeatures(
            features=self.hidden_features, **self.hidden_kwargs,
        )
        self.sngp_dense_layer = nn.Dense(features=self.features, **self.output_kwargs)
        self.sngp_covariance_layer = LaplaceRandomFeatureCovariance(
            hidden_features=self.hidden_features, **self.covariance_kwargs
        )
        # pylint:enable=invalid-name,not-a-mapping

    def __call__(
        self,
        inputs: Array,
        return_full_covariance: bool = False,
    ) -> Tuple[Array, Array]:
        """
        Computes Gaussian process outputs.

        Parameters
        ----------
        inputs: Array
            The nd-array of shape (batch_size, ..., input_dim).
        return_full_covariance: bool
            Whether to return the full covariance matrix, shape
            (batch_size, batch_size), or only return the predictive variances with
            shape (batch_size, ).

        Returns
        -------
        Tuple[Array, Array]
          A tuple of predictive logits, predictive covariance and (optionally)
          random Fourier features.
        """
        gp_inputs = self.sngp_norm_layer(inputs) if self.normalize_input else inputs
        gp_features = self.sngp_random_features_layer(gp_inputs)

        gp_logits = self.sngp_dense_layer(gp_features)
        gp_covariance = self.sngp_covariance_layer(
            gp_features, gp_logits, diagonal_only=not return_full_covariance
        )
        return gp_logits, gp_covariance


class RandomFourierFeatures(nn.Module):
    """
    A random fourier feature (RFF) layer that approximates a kernel model.

    The random feature transformation is a one-hidden-layer network with
    non-trainable weights (see, e.g., Algorithm 1 of
    `Random Features for Large-Scale
    Kernel Machines <https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf>`_):

    .. math::
        f(x) = \gamma * cos(\mathbf{W}\mathbf{x} + \mathbf{b})

    where :math:`\mathbf{W}` is the kernel matrix, :math:`\mathbf{b}` is the bias
    and :math:`\gamma` is the output scale.
    The forward pass logic closely follows that of the `nn.Dense` layer.

    Attributes
    ----------
    features: int
        The number of output units.
    feature_scale: Optional[float]
        Scale to apply to the output.
        When using GP layer as the output layer of a nerual network, it is recommended to set this to 1.
        to prevent it from changing the learning rate to the hidden layers.
    kernel_init:  Callable[[PRNGKeyArray, Shape, Type], Array]
         Callable[[PRNGKeyArray, Shape, Type], Array] function for the weight matrix.
    bias_init:  Callable[[PRNGKeyArray, Shape, Type], Array]
         Callable[[PRNGKeyArray, Shape, Type], Array] function for the bias.
    seed: int
        Random seed for generating random features. This will override the external RNGs.
    dtype: Type
        The dtype of the computation.
    """
    features: int
    kernel_scale: Optional[float] = 1.0
    feature_scale: Optional[float] = 1.0
    kernel_init:  Callable[[PRNGKeyArray, Shape, Type], Array] = default_rbf_kernel_init
    bias_init:  Callable[[PRNGKeyArray, Shape, Type], Array] = default_rbf_bias_init
    seed: int = 0
    dtype: Type = jnp.float32
    collection_name: str = "random_features"

    def setup(self):
        # Defines the random number generator.
        self.rng = random.PRNGKey(self.seed)

        # Processes random feature scale.
        self._feature_scale = self.feature_scale
        if self._feature_scale is None:
            self._feature_scale = jnp.sqrt(2.0 / self.features)
        self._feature_scale = jnp.asarray(self._feature_scale, dtype=self.dtype)

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        """
        Applies random feature transformation along the last dimension of inputs.

        Parameters
        ----------
        inputs: Array
            The nd-array to be transformed.

        Returns
        -------
        Array
          The transformed input.
        """
        # Initializes variables.
        input_dim = inputs.shape[-1]

        kernel_rng, bias_rng = random.split(self.rng, num=2)
        kernel_shape = (input_dim, self.features)

        kernel = self.variable(
            self.collection_name,
            "kernel",
            self.kernel_init,
            kernel_rng,
            kernel_shape,
            self.dtype,
        )
        kernel_scale = self.param(
            "rf_kernel_scale",
            nn.initializers.constant(self.kernel_scale),
            (1,),
            self.dtype,
        )
        bias = self.variable(
            self.collection_name,
            "bias",
            self.bias_init,
            bias_rng,
            (self.features,),
            self.dtype,
        )

        # Specifies multiplication dimension.
        contracting_dims = ((inputs.ndim - 1,), (0,))
        batch_dims = ((), ())

        # Performs forward pass.
        inputs = jnp.asarray(inputs, self.dtype)

        outputs = lax.dot_general(inputs, (1. / kernel_scale) * kernel.value, (contracting_dims, batch_dims))
        outputs = outputs + jnp.broadcast_to(bias.value, outputs.shape)

        return self._feature_scale * jnp.cos(outputs)


class LaplaceRandomFeatureCovariance(nn.Module):
    """
    Computes the approximated posterior covariance using Laplace method.

    Attributes
    ----------
    hidden_features: int
        The number of random fourier features.
    ridge_penalty: float
        Initial Ridge penalty to weight covariance matrix.
        This value is used to stablize the eigenvalues of weight covariance estimate :math:`\Sigma` so that
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
    dtype: Type
        The dtype of the computation
    """

    hidden_features: int
    ridge_penalty: float = 1.0
    momentum: Optional[float] = None
    collection_name: str = "laplace_covariance"
    dtype: Type = jnp.float32

    def setup(self):
        if self.momentum is not None:
            if self.momentum < 0.0 or self.momentum > 1.0:
                raise ValueError(
                    f"`momentum` must be between (0, 1). " f"Got {self.momentum}."
                )

    @nn.compact
    def __call__(
        self,
        gp_features: Array,
        gp_logits: Optional[Array] = None,
        diagonal_only: bool = True,
    ) -> Optional[Array]:
        """
        Updates the precision matrix and computes the predictive covariance.

        NOTE:
        The precision matrix will be updated only during training (i.e., when
        `self.collection_name` are in the list of mutable variables). The covariance
        matrix will be computed only during inference to avoid repeated calls to the
        (expensive) `linalg.inv` op.

        Parameters
        ----------
        gp_features: Array
            The nd-array of random fourier features, shape (batch_size, ..., hidden_features).
        gp_logits: Optional[Array]
            The nd-array of predictive logits, shape (batch_size, ..., logit_dim).
            Cannot be None.
        diagonal_only: bool
            Whether to return only the diagonal elements of the predictive covariance matrix (i.e., the predictive variance).

        Returns
        -------
        Optional[Array]
          The predictive variances of shape (batch_size, ) if diagonal_only=True,
          otherwise the predictive covariance matrix of shape
          (batch_size, batch_size).
        """
        gp_features = jnp.asarray(gp_features, self.dtype)

        # Flatten GP features and logits to 2-d, by doing so we treat all the
        # non-final dimensions as the batch dimensions.
        gp_features = jnp.reshape(gp_features, [-1, self.hidden_features])

        if gp_logits is not None:
            gp_logits = jnp.asarray(gp_logits, self.dtype)
            gp_logits = jnp.reshape(gp_logits, [gp_features.shape[0], -1])

        precision_matrix = self.variable(
            self.collection_name,
            "precision_matrix",
            lambda: self.initial_precision_matrix(),
        )  # pylint: disable=unnecessary-lambda

        # Updates the precision matrix during training.
        initializing = self.is_mutable_collection("params")
        training = self.is_mutable_collection(self.collection_name)

        if training and not initializing:
            precision_matrix.value = self.update_precision_matrix(
                gp_features, gp_logits, precision_matrix.value
            )

        # Computes covariance matrix during inference.
        if not training:
            return self.compute_predictive_covariance(
                gp_features, precision_matrix, diagonal_only
            )

    def initial_precision_matrix(self):
        """Returns the initial diagonal precision matrix."""
        return jnp.eye(self.hidden_features, dtype=self.dtype) * self.ridge_penalty

    def update_precision_matrix(
        self, gp_features: Array, gp_logits: Optional[Array], precision_matrix: Array
    ) -> Array:
        """Updates precision matrix given a new batch.

        Parameters
        ----------
          gp_features: Array
            Random features from the new batch, shape (batch_size, hidden_features)
          gp_logits: Optional[Array]
            Predictive logits from the new batch, shape (batch_size, logit_dim).
            Currently only `logit_dim=1` is supported.
          precision_matrix: Array
            The current precision matrix, shape (hidden_features, hidden_features).

        Returns
        -------
        Array
            Updated precision matrix, shape (hidden_features, hidden_features).
        """
        # Computes precision matrix within new batch.
        prob_multiplier = 1.0

        gp_features_adj = jnp.sqrt(prob_multiplier) * gp_features
        batch_prec_mat = jnp.matmul(jnp.transpose(gp_features_adj), gp_features_adj)

        # Updates precision matrix.
        if self.momentum is None:
            # Performs exact update without momentum.
            precision_matrix_updated = precision_matrix + batch_prec_mat
        else:
            batch_size = gp_features.shape[0]
            precision_matrix_updated = (
                self.momentum * precision_matrix
                + (1 - self.momentum) * batch_prec_mat / batch_size
            )
        return precision_matrix_updated

    def compute_predictive_covariance(
        self, gp_features: Array, precision_matrix: nn.Variable, diagonal_only: bool
    ) -> Array:
        """
        Computes the predictive covariance.

        Approximates the Gaussian process posterior using random features.
        Given training random feature :math:`\mathbf{\Phi_{tr}}` (num_train, num_hidden) and testing
        random feature :math:`\mathbf{\Phi_{ts}}` (batch_size, num_hidden). The predictive covariance
        matrix is computed as (assuming Gaussian likelihood):
        :math:`s * \mathbf{\Phi_{ts}}(\mathbf{I}*s + \mathbf{\Phi_{tr}}^{T}*\mathbf{\Phi_{tr}})^{-1}\mathbf{\Phi_{tr}}^{^T}`

        where :math:`s` is the ridge factor to be used for stablizing the inverse, and \mathbf{I} is
        the identity matrix with shape (num_hidden, num_hidden). The above
        description is formal only: the actual implementation uses a Cholesky
        factorization of the covariance matrix.

        Parameters
        ----------
        gp_features: Array
            The random feature of testing data to be used for computing the covariance matrix.
            Shape (batch_size, gp_hidden_size).
        precision_matrix: nn.Variable
            The model's precision matrix.
        diagonal_only: bool
            Whether to return only the diagonal elements of the predictive covariance matrix
            (i.e., the predictive variances).

        Returns
        -------
        Array
            The predictive variances of shape (batch_size, ) if `diagonal_only=True`,
            otherwise the predictive covariance matrix of shape (batch_size, batch_size).
        """
        chol = linalg.cholesky(precision_matrix.value)
        chol_t_cov_feature_product = linalg.triangular_solve(
            chol, gp_features.T, left_side=True, lower=True
        )

        if diagonal_only:
            # Compute diagonal element only, shape (batch_size, ).
            gp_covar = jnp.square(chol_t_cov_feature_product).sum(0)
        else:
            # Compute full covariance matrix, shape (batch_size, batch_size).
            gp_covar = chol_t_cov_feature_product.T @ chol_t_cov_feature_product

        return self.ridge_penalty * gp_covar