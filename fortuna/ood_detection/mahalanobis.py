import logging
from typing import Tuple

import jax
import jax.numpy as jnp
import tqdm

from fortuna.data import BaseInputsLoader
from fortuna.data.loader.base import BaseDataLoaderABC
from fortuna.ood_detection.base import (
    NotFittedError,
    OutOfDistributionClassifierABC,
)
from fortuna.prob_model.posterior.state import PosteriorState


@jax.jit
def compute_mean_and_joint_cov(
    embeddings: jnp.ndarray, labels: jnp.ndarray, class_ids: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Computes class-specific means and a shared covariance matrix given the training set embeddings
    (e.g., the last-layer representation of the model for each training example).

    Parameters
    ----------
    embeddings: jnp.ndarray
        An array of shape `(n, d)`, where `n` is the sample size of training set,
        `d` is the dimension of the embeddings.
    labels: jnp.ndarray
        An array of shape `(n,)`
    class_ids: jnp.ndarray
        An array of the unique class ids in `labels`.

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray]
        A tuple containing:
        1) A `jnp.ndarray` of len n_class, and the i-th element is an np.array of size
      ` (d,)` corresponding to the mean of the fitted Gaussian distribution for the i-th class;
        2) The shared covariance matrix of shape `(d, d)`.
    """
    n_dim = embeddings.shape[1]
    cov = jnp.zeros((n_dim, n_dim))

    def f(cov, class_id):
        mask = jnp.expand_dims(labels == class_id, axis=-1)
        data = embeddings * mask
        mean = jnp.sum(data, axis=0) / jnp.sum(mask)
        diff = (data - mean) * mask
        cov += jnp.matmul(diff.T, diff)
        return cov, mean

    cov, means = jax.lax.scan(f, cov, class_ids)
    cov = cov / len(labels)
    return means, cov


@jax.jit
def compute_mahalanobis_distance(
    embeddings: jnp.ndarray, means: jnp.ndarray, cov: jnp.ndarray
) -> jnp.ndarray:
    """
    Computes Mahalanobis distance between the input and the fitted Guassians.

    Parameters
    ----------
    embeddings: jnp.ndarray
        A matrix of shape `(n, d)`, where `n` is the sample size of the test set, and `d` is the size of the embeddings.
    means: jnp.ndarray
        A matrix of shape `(c, d)`, where `c` is the number of classes in the classification task.
        The ith row of the matrix corresponds to the mean of the fitted Gaussian distribution for the i-th class.
    cov: jnp.ndarray
        The shared covariance mmatrix of the shape `(d, d)`.

    Returns
    -------
    A matrix of size `(n, c)` where the `(i, j)` element
    corresponds to the Mahalanobis distance between i-th sample to the j-th
    class Gaussian.
    """
    # NOTE: It's possible for `cov` to be singular, in part because it is
    # estimated on a sample of data. This can be exacerbated by lower precision,
    # where, for example, the matrix could be non-singular in float64, but
    # singular in float32. For our purposes in computing Mahalanobis distance,
    # using a  pseudoinverse is a reasonable approach that will be equivalent to
    # the inverse if `cov` is non-singular.
    cov_inv = jnp.linalg.pinv(cov)

    def maha_dist(x, mean):
        # NOTE: This computes the squared Mahalanobis distance.
        diff = x - mean
        return jnp.einsum("i,ij,j->", diff, cov_inv, diff)

    maha_dist_all_classes_fn = jax.vmap(maha_dist, in_axes=(None, 0))
    out = jax.lax.map(lambda x: maha_dist_all_classes_fn(x, means), embeddings)
    return out


class MalahanobisClassifierABC(OutOfDistributionClassifierABC):
    """
    The pre-trained features of a softmax neural classifier :math:`f(\mathbf{x})` are assumed to follow a
    class-conditional gaussian distribution with a tied covariance matrix :math:`\mathbf{\Sigma}`:

    .. math::
        P(f(\mathbf{x})|y=k) = \mathcal{N}(f(\mathbf{x})|\mu_k, \mathbf{\Sigma})

    for all :math:`k \in {1,...,K}`, where :math:`K` is the number of classes.

    The confidence score :math:`M(\mathbf{x})` for a new test sample :math:`\mathbf{x}` is obtained computing
    the max (squared) Mahalanobis distance between :math:`f(\mathbf{x})` and the fitted class-wise guassians.

    See `Lee, Kimin, et al. <https://proceedings.neurips.cc/paper/2018/file/abdeb6f575ac5c6676b747bca8d09cc2-Paper.pdf>`_
    """

    def __init__(self, *args, **kwargs):
        super(MalahanobisClassifierABC, self).__init__(*args, **kwargs)
        self._maha_dist_all_classes_fn = None

    def fit(
        self,
        state: PosteriorState,
        train_data_loader: BaseDataLoaderABC,
        num_classes: int,
    ) -> None:
        """
        Fits a Multivariate Gaussian to the training data using class-specific means and a shared covariance matrix.

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
        for x, y in tqdm.tqdm(
            train_data_loader, desc="Computing embeddings for Malhanbis Classifier: "
        ):
            train_embeddings.append(
                self.apply(inputs=x, params=state.params, mutable=state.mutable)
            )
            train_labels.append(y)
        train_embeddings = jnp.concatenate(train_embeddings, 0)
        train_labels = jnp.concatenate(train_labels)

        n_labels_observed = len(jnp.unique(train_labels))
        if n_labels_observed != num_classes:
            logging.warning(
                f"{num_classes} labels were expected but found {n_labels_observed} in the train set. "
                f"Will proceed but performance may be hurt by this."
            )

        means, cov = compute_mean_and_joint_cov(
            train_embeddings, train_labels, jnp.arange(num_classes)
        )
        self._maha_dist_all_classes_fn = lambda x: compute_mahalanobis_distance(
            x, means, cov
        )

    def score(
        self, state: PosteriorState, inputs_loader: BaseInputsLoader
    ) -> jnp.ndarray:
        """
        The confidence score :math:`M(\mathbf{x})` for a new test sample :math:`\mathbf{x}` is obtained computing
        the max (squared) Mahalanobis distance between :math:`f(\mathbf{x})` and the fitted class-wise guassians.

        A high score signals that the test sample :math:`\mathbf{x}` is identified as OOD.

        Parameters
        ----------
        state: PosteriorState
            The posterior state of a pre-trained model
        inputs_loader:  BaseInputsLoader
            The inputs loader (data only, no labels)

        Returns
        -------
        jnp.ndarray
            An array with scores for each sample in `inputs_loader`.
        """
        if self._maha_dist_all_classes_fn is None:
            raise NotFittedError("You have to call fit before calling score.")
        embeddings = jnp.concatenate(
            [
                self.apply(inputs=x, params=state.params, mutable=state.mutable)
                for x in inputs_loader
            ],
            0,
        )
        return self._maha_dist_all_classes_fn(embeddings).min(axis=1)
