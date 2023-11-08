from typing import (
    Callable,
    Iterable,
    Optional,
)

import numpy as np
from tqdm import tqdm

from fortuna.typing import Array


class EmbeddingManager:
    def __init__(
        self,
        encoding_fn: Callable[[Array], Array],
        reduction_fn: Optional[Callable[[Array], Array]] = None,
    ):
        self.encoding_fn = encoding_fn
        self.reduction_fn = reduction_fn

    def embed(self, inputs: Iterable) -> Array:
        """
        Embed the inputs by first applying a reduction function, if available, and then an encoding function.

        Parameters
        ----------
        inputs: Iterable
            An iterable of inputs.

        Returns
        -------
        Array
            The embeddings.
        """
        embeddings = []
        for x in tqdm(inputs, desc="Batch"):
            embeddings.append(self.encoding_fn(x).tolist())
        embeddings = np.concatenate(embeddings, axis=0)
        if self.reduction_fn is not None:
            embeddings = self.reduction_fn(embeddings)
        return embeddings
