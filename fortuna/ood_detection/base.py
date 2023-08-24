import abc

from fortuna.typing import Array


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
