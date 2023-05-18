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
from fortuna.typing import InputData, Params, Mutable


class NotFittedError(ValueError, AttributeError):
    """Exception class to raise if estimator is used before fitting."""


class OutOfDistributionClassifierABC:
    """
    Post-training classifier that uses the training sample embeddings coming from the model
    to score a (new) test sample w.r.t. its chance of belonging to the original training distribution
    (i.e, it is in-distribution) or not (i.e., it is out of distribution).
    """

    def __init__(self, feature_extractor_subnet: nn.Module):
        """
        Parameters
        ----------
        feature_extractor_subnet: nn.Module
            The model (or a part of it) used to obtain the embeddings of any given input.
        """
        self.feature_extractor_subnet = feature_extractor_subnet

    @abc.abstractmethod
    def apply(
        self,
        inputs: InputData,
        params: Params,
        mutable: Mutable,
        **kwargs,
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, PyTree]]:
        """
        Transform an input :math:`\mathbf{x}` into an embedding :math:`f(\mathbf{x})`.
        """
        pass
        # return self.feature_extractor_subnet(**inputs, train=False)[1]

    @abc.abstractmethod
    def fit(
        self,
        state: PosteriorState,
        train_data_loader: BaseDataLoaderABC,
        num_classes: int,
    ) -> None:
        pass

    @abc.abstractmethod
    def score(
        self, state: PosteriorState, inputs_loader: BaseInputsLoader
    ) -> jnp.ndarray:
        pass
