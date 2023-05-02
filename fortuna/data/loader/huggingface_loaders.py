from typing import Union, Dict, Iterable, Optional, Tuple
from jax import numpy as jnp

from fortuna.data.loader.base import BaseDataLoaderABC, BaseInputsLoader, BaseTargetsLoader
from fortuna.data.loader.utils import IterableData
from fortuna.typing import Array


class HuggingFaceDataLoader(BaseDataLoaderABC):

    def __init__(
            self,
            iterable: Union[Iterable[Dict[str, Array]], Iterable[Tuple[Dict[str, Array], Array]]],
            num_unique_labels: int = None,
            num_input_samples: Optional[int] = None
    ):
        """
        A data loader class.

        Parameters
        ----------
        iterable : Union[Iterable[Dict[str, Array]], Iterable[Tuple[Dict[str, Array],Array]]]
            A data loader obtained via :func:`~HuggingFaceClassificationDataset.get_dataloader`.
        """
        super().__init__(iterable, num_unique_labels)
        self._num_input_samples = num_input_samples

    @property
    def size(self):
        if self._num_input_samples:
            return self._num_input_samples
        else:
            return super().size

    @property
    def num_unique_labels(self) -> Optional[int]:
        if self._num_unique_labels is None:
            self._num_unique_labels = len(jnp.unique(self.to_array_targets()))
        return self._num_unique_labels

    def to_array_targets(self):
        targets = []
        for (_, batch_targets) in self:
            targets.append(batch_targets)
        return jnp.concatenate(targets, 0)

    def to_inputs_loader(self):
        return BaseInputsLoader(IterableData.data_loader_to_inputs_iterable(self))

    def to_targets_loader(self):
        return BaseTargetsLoader(IterableData.data_loader_to_targets_iterable(self))


