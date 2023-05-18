"""
Adapted from https://github.com/omegafragger/DDU/blob/main/utils/gmm_utils.py
"""
from typing import (
    Callable,
    Tuple,
)

import jax
from jax import numpy as jnp
import jax.scipy as jsp
import jax.scipy.stats as jsp_stats
import numpy as np
from tqdm import tqdm

from fortuna.data import BaseInputsLoader
from fortuna.data.loader.base import BaseDataLoaderABC
from fortuna.ood_detection.base import (
    NotFittedError,
    OutOfDistributionClassifierABC,
)
from fortuna.prob_model.posterior.state import PosteriorState
from fortuna.typing import Array

DOUBLE_INFO = np.finfo(np.double)
JITTERS = [0, DOUBLE_INFO.tiny] + [10**exp for exp in range(-10, 0, 1)]


def _centered_cov(x: Array) -> Array:
    n = x.shape[0]
    res = jnp.matmul(1 / (n - 1) * x.T, x)
    return res


def compute_classwise_mean_and_cov(
    embeddings: Array, labels: Array, num_classes: int
) -> Tuple[Array, Array]:
    """
    Computes class-specific means and a covariance matrices given the training set embeddings
    (e.g., the last-layer representation of the model for each training example).

    Parameters
    ----------
    embeddings: Array
        The embeddings of shape `(n, d)` where `n` is the number of training samples and `d` is the embbeding's size.
    labels: Array
        An array of length `n` containing, for each input sample, its ground-truth label.
    num_classes: int
        The total number of classes available in the classification task.

    Returns
    ----------
    Tuple[Array, Array]:
        A tuple containing:
        1) an `Array` containing the per-class mean vector of the fitted GMM.
        The shape of the array is `(num_classes, d)`.
        2) an `Array` containing the per-class covariance matrix of the fitted GMM.
        The shape of the array is `(num_classes, d, d)`.
    """
    #
    classwise_mean_features = np.stack(
        [jnp.mean(embeddings[labels == c], 0) for c in range(num_classes)]
    )
    #
    classwise_cov_features = np.stack(
        [
            _centered_cov(embeddings[labels == c] - classwise_mean_features[c])
            for c in range(num_classes)
        ]
    )
    return classwise_mean_features, classwise_cov_features


def _get_logpdf_fn(
    classwise_mean_features: Array, classwise_cov_features: Array
) -> Callable[[Array], Array]:
    """
    Returns a function to evaluate the log-likelihood of a test sample according to the (fitted) GMM.

    Parameters
    ----------
    classwise_mean_features: Array
        The per-class mean vector of the fitted GMM. The shape of the array is `(num_classes, d)`.
    classwise_cov_features: Array
        The per-class covariance matrix of the fitted GMM. The shape of the array is `(num_classes, d, d)`.

    Returns
    -------
    Callable[[Array], Array]
        A function to evaluate the log-likelihood of a test sample according to the (fitted) GMM.
    """
    for jitter_eps in JITTERS:
        jitter = np.expand_dims(jitter_eps * np.eye(classwise_cov_features.shape[1]), 0)
        gmm_logprob_fn_vmapped = jax.vmap(
            jsp_stats.multivariate_normal.logpdf, in_axes=(None, 0, 0)
        )
        gmm_logprob_fn = lambda x: gmm_logprob_fn_vmapped(
            x, classwise_mean_features, (classwise_cov_features + jitter)
        ).T

        nans = np.isnan(gmm_logprob_fn(classwise_mean_features)).sum()
        if nans > 0:
            print(f"Nans, jittering {jitter_eps}")
            continue
        break

    return gmm_logprob_fn


class DeepDeterministicUncertaintyABC(OutOfDistributionClassifierABC):
    """
    A Gaussian Mixture Model :math:`q(\mathbf{x}, z)` with a single Gaussian mixture component per class :math:`k \in {1,...,K}`
    is fit after training.
    Each class component is fit computing the empirical mean :math:`\mathbf{\hat{\mu}_k}` and covariance matrix
    :math:`\mathbf{\hat{\Sigma}_k}` of the feature vectors :math:`f(\mathbf{x})`.

    The confidence score :math:`M(\mathbf{x})` for a new test sample is obtained computing the negative marginal likelihood
    of the feature representation.

    See `Mukhoti, Jishnu, et al. <https://arxiv.org/abs/2102.11582>`_
    """

    def __init__(self, *args, **kwargs):
        super(DeepDeterministicUncertaintyABC, self).__init__(*args, **kwargs)
        self._gmm_logpdf_fn = None

    def fit(
        self,
        state: PosteriorState,
        train_data_loader: BaseDataLoaderABC,
        num_classes: int,
    ) -> None:
        """
        Fits a Multivariate Gaussian to the training data using class-specific means and covariance matrix.

        Parameters
        ----------
        state: PosteriorState
            the posterior state ob a pre-trained model
        train_data_loader: BaseDataLoaderABC
            the training data loader (covariates and target)
        num_classes: int
            the number of classes for the training task
        """
        train_labels = []
        train_embeddings = []
        for x, y in tqdm(train_data_loader, desc="Computing embeddings for DDU: "):
            train_embeddings.append(
                self.apply(inputs=x, params=state.params, mutable=state.mutable)
            )
            train_labels.append(y)
        train_embeddings = jnp.concatenate(train_embeddings, 0)
        train_labels = jnp.concatenate(train_labels)

        (
            classwise_mean_features,
            classwise_cov_features,
        ) = compute_classwise_mean_and_cov(train_embeddings, train_labels, num_classes)
        self._gmm_logpdf_fn = _get_logpdf_fn(
            classwise_mean_features, classwise_cov_features
        )

    def score(self, state: PosteriorState, inputs_loader: BaseInputsLoader) -> Array:
        """
        The confidence score :math:`M(\mathbf{x})` for a new test sample :math:`\mathbf{x}` is obtained computing
        the negative marginal likelihood of the feature representation
        :math:`-q(f(\mathbf{x})) = - \sum\limits_{k}q(f(\mathbf{x})|y) q(y)`.

        A high score signals that the test sample :math:`\mathbf{x}` is identified as OOD.

        Parameters
        ----------
        state: PosteriorState
            The posterior state of a pre-trained model
        inputs_loader:  BaseInputsLoader
            The inputs loader (data only, no labels)

        Returns
        -------
        Array
            An array with scores for each sample in `inputs_loader`.
        """
        if self._gmm_logpdf_fn is None:
            raise NotFittedError("You have to call fit before calling score.")
        embeddings = jnp.concatenate(
            [
                self.apply(inputs=x, params=state.params, mutable=state.mutable)
                for x in inputs_loader
            ],
            0,
        )
        loglik = self._gmm_logpdf_fn(embeddings)
        return -jsp.special.logsumexp(jnp.nan_to_num(loglik, 0.0), axis=1)
