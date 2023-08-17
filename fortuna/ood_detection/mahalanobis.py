import logging
from typing import Tuple

import jax
import jax.numpy as jnp

from fortuna.ood_detection.base import (
    NotFittedError,
    OutOfDistributionClassifierABC,
)
from fortuna.typing import Array


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


class MalahanobisOODClassifier(OutOfDistributionClassifierABC):
    """
    The pre-trained features of a softmax neural classifier :math:`f(\mathbf{x})` are assumed to follow a
    class-conditional gaussian distribution with a tied covariance matrix :math:`\mathbf{\Sigma}`:

    .. math::
        \mathbb{P}(f(\mathbf{x})|y=k) = \mathcal{N}(f(\mathbf{x})|\mu_k, \mathbf{\Sigma})

    for all :math:`k \in {1,...,K}`, where :math:`K` is the number of classes.

    The confidence score :math:`M(\mathbf{x})` for a new test sample :math:`\mathbf{x}` is obtained computing
    the max (squared) Mahalanobis distance between :math:`f(\mathbf{x})` and the fitted class-wise guassians.

    See `Lee, Kimin, et al. <https://proceedings.neurips.cc/paper/2018/file/abdeb6f575ac5c6676b747bca8d09cc2-Paper.pdf>`_
    """

    def __init__(self, *args, **kwargs):
        super(MalahanobisOODClassifier, self).__init__(*args, **kwargs)
        self._mean = None
        self._cov = None

    @property
    def mean(self):
        """
        Returns
        -------
        Array
            A matrix of shape `(num_classes, d)`, where `num_classes` is the number of classes in the in-distribution
            classification task.
            The ith row of the matrix corresponds to the mean of the fitted Gaussian distribution for the i-th class.
        """
        return self._mean

    @property
    def cov(self):
        """
         Returns
        -------
        Array
            The shared covariance matrix with shape `(d, d)`, where `d` is the embedding size.
        """
        return self.cov

    def fit(self, embeddings: Array, targets: Array) -> None:
        """
        Fits a Multivariate Gaussian to the training data using class-specific means and a shared covariance matrix.

        Parameters
        ----------
        embeddings: Array
            The embeddings of shape `(n, d)` where `n` is the number of training samples and `d` is the embbeding's size.
        targets: Array
            An array of length `n` containing, for each input sample, its ground-truth label.
        """
        n_labels_observed = len(jnp.unique(targets))
        if n_labels_observed != self.num_classes:
            logging.warning(
                f"{self.num_classes} labels were expected but found {n_labels_observed} in the provided train set. "
                f"Will proceed but performance may be hurt by this."
            )

        self._mean, self._cov = compute_mean_and_joint_cov(
            embeddings, targets, jnp.arange(self.num_classes)
        )

    def score(self, embeddings: Array) -> Array:
        """
        The confidence score :math:`M(\mathbf{x})` for a new test sample :math:`\mathbf{x}` is obtained computing
        the max (squared) Mahalanobis distance between :math:`f(\mathbf{x})` and the fitted class-wise Guassians.

        A high score signals that the test sample :math:`\mathbf{x}` is identified as OOD.

        Parameters
        ----------
        embeddings: Array
            The embeddings of shape `(n, d)` where `n` is the number of test samples and `d` is the embbeding's size.

        Returns
        -------
        Array
            An array of scores with length `n`.
        """
        if self._mean is None or self._cov is None:
            raise NotFittedError("You have to call fit before calling score.")
        return compute_mahalanobis_distance(embeddings, self.mean, self.cov).min(axis=1)
