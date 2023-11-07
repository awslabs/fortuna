from typing import Callable

import numpy as np
from tqdm import (
    tqdm,
    trange,
)

from fortuna.data import InputsLoader
from fortuna.typing import Array


class EmbeddingManager:
    def __init__(
        self,
        encoding_fn: Callable[[Array], Array],
        reduction_fn: Callable[[Array], Array],
    ):
        self.encoding_fn = encoding_fn
        self.reduction_fn = reduction_fn

    def get(self, inputs_loader: InputsLoader) -> Array:
        embeddings = []
        for inputs in tqdm(inputs_loader, desc="Batch"):
            embeddings.append(
                np.vstack(
                    [
                        self.encoding_fn(inputs[i]).tolist()
                        for i in trange(len(inputs), desc="Encode")
                    ]
                )
            )
        embeddings = np.concatenate(embeddings, axis=0)
        embeddings = self.reduction_fn(embeddings)
        return embeddings
