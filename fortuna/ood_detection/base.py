import abc
from functools import partial
from typing import (
    Tuple,
    Union,
)

from flax import linen as nn
from flax.training.checkpoints import PyTree
import jax
from jax import numpy as jnp

from fortuna.data.loader.base import (
    BaseDataLoaderABC,
    BaseInputsLoader,
)
from fortuna.prob_model.posterior.state import PosteriorState
from fortuna.typing import InputData, Params, Mutable, Array


class NotFittedError(ValueError, AttributeError):
    """Exception class to raise if estimator is used before fitting."""


class OutOfDistributionClassifierABC:
    """
    Post-training classifier that uses the training sample embeddings coming from the model
    to score a (new) test sample w.r.t. its chance of belonging to the original training distribution
    (i.e, it is in-distribution) or not (i.e., it is out of distribution).
    """

    def __init__(self, num_classes: int):
        """
        Parameters
        ----------
        num_classes: int
            The number of classes for the in-distribution classification task.
        """
        self.num_classes = num_classes

    @abc.abstractmethod
    def fit(self, embeddings: Array, targets: Array) -> None:
        pass

    @abc.abstractmethod
    def score(self, embeddings: Array) -> Array:
        pass
