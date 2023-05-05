import abc
import logging

from typing import Union, Dict, List, Iterable, Optional, Tuple, Sequence, Any
from tqdm import tqdm
import jax.random
from datasets import DatasetDict, Dataset
from jax import numpy as jnp
from jax.random import PRNGKeyArray
from transformers import PreTrainedTokenizer, BatchEncoding, DataCollatorWithPadding, DataCollatorForLanguageModeling
from transformers.utils import PaddingStrategy

from fortuna.data.loader.huggingface_loaders import HuggingFaceDataLoader
from fortuna.data.loader.utils import IterableData
from fortuna.typing import Array

logger = logging.getLogger(__name__)


class HuggingFaceClassificationDatasetABC(abc.ABC):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        padding: Union[bool, str, PaddingStrategy] = True,
        num_unique_labels: Optional[int] = None,
    ):
        """
        A Dataset class to work with HuggingFace-like datasets.

        Parameters
        ----------
        tokenizer: PreTrainedTokenizer
            A pretrained tokenizer
        max_length: int
            Maximum sequence length (equal for each batch).
            It should be se to max length accepted by the model or a smaller number.
        padding: Union[bool, str, PaddingStrategy]
            See `Padding and Truncation <https://huggingface.co/docs/transformers/pad_truncation>`_
            for more information (`truncation` is always True).
        num_unique_labels: Optional[int]
            Number of unique target labels in the task (classification only)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.num_unique_labels = num_unique_labels
        self._data_collator = None

    @abc.abstractmethod
    def get_tokenized_datasets(self, datasets: DatasetDict, text_columns: Sequence[str] = ('sentence',), target_column: str = 'label') -> DatasetDict:
        """
        Converts one or more dataset into sequences of tokens, using the tokenizer.

        Parameters
        ----------
        datasets: DatasetDict
            A dictionary of `datasets.Dataset` that have to be encoded.
        text_columns: str
            A list containing the text column names, whose text sequences has to be encoded.
        target_column: str
            The target column name

        Returns
        -------
        DatasetDict
            A dictionary of tokenized datasets.
        """
        pass

    @property
    @abc.abstractmethod
    def data_collator(self):
        """
        Create a data collator object that will process batches of data and apply transformation on them if needed.
        """
        pass

    def get_data_loader(self, dataset: Dataset,  per_device_batch_size: int, rng: PRNGKeyArray, shuffle: bool = False, drop_last: bool = False, verbose: bool = False) -> HuggingFaceDataLoader:
        """
        Build a :class:`~fortuna.data.loader.huggingface_loaders.HuggingFaceDataLoader` object from a
        tokenized dataset.

        Parameters
        ----------
        dataset: Dataset
            A tokenizeed dataset (see :meth:`.HuggingFaceClassificationDatasetABC.get_tokenized_datasets`).
        per_device_batch_size: bool
            Batch size for each device.
        rng: PRNGKeyArray
            Random number generator.
        shuffle: bool
            if True, shuffle the data so that each batch is a ranom sample from the dataset.
        drop_last: bool
            if True, the last batch (which potentially is smaller then the default batch size) is dropped.
        verbose: bool
            Whether to show a progress bar while iterating over the dataloader or not.

        Returns
        -------
        HuggingFaceDataLoader
            The dataloader
        """
        iterable = IterableData.from_callable(
            lambda *args, **kwargs: self._get_data_loader(
                dataset,
                batch_size=per_device_batch_size * jax.local_device_count(),
                shuffle=shuffle,
                drop_last=drop_last,
                rng=rng,
                verbose=verbose,
            )
        )
        return HuggingFaceDataLoader(
            iterable=iterable,
            num_inputs=len(dataset),
            num_unique_labels=self.num_unique_labels
        )

    @abc.abstractmethod
    def _collate(self, batch: Dict[str, Array], batch_size: int) -> Dict[str, Array]:
        pass

    @staticmethod
    def _get_batches_idxs(rng: PRNGKeyArray, dataset_size: int, batch_size: int, shuffle: bool = False, drop_last: bool = False) -> Iterable[Array]:
        if shuffle:
            dataset_idxs = jax.random.permutation(rng, dataset_size)  # batch idxs
        else:
            dataset_idxs = jnp.arange(dataset_size)
        if drop_last:
            steps_per_epoch = dataset_size // batch_size
            dataset_idxs = dataset_idxs[: steps_per_epoch * batch_size]  # Skip incomplete batch.
        yield from [dataset_idxs[start:start+batch_size] for start in range(0, len(dataset_idxs), batch_size)]

    def _get_data_loader(
            self, dataset: Dataset,  batch_size: int, shuffle: bool = False, drop_last: bool = False,
            rng: Optional[PRNGKeyArray] = None, verbose: bool = False,
    ) -> Union[Iterable[Dict[str, Array]], Iterable[Tuple[Dict[str, Array],Array]]]:
        rng = jax.random.split(rng, 1)[0]

        batch_idxs_gen = self._get_batches_idxs(
            rng=rng,
            dataset_size=len(dataset),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        if verbose:
            steps_per_epoch = len(dataset) // batch_size
            steps_per_epoch = steps_per_epoch if drop_last else steps_per_epoch + 1
            batch_idxs_gen = tqdm(batch_idxs_gen, desc="Batches:", total=steps_per_epoch)
        for batch_idxs in batch_idxs_gen:
            batch = dataset[batch_idxs]
            batch_labels = None
            if 'labels' in batch:
                batch_labels = jnp.array(batch.pop('labels'))
            batch = self._collate(batch, len(batch['input_ids']))
            batch_inputs = {k: jnp.array(v) for k, v in batch.items()}
            if batch_labels is not None:
                yield (batch_inputs, batch_labels)
            else:
                yield batch_inputs, None


class HuggingFaceClassificationDatasetForSequenceClassification(HuggingFaceClassificationDatasetABC):
    def get_tokenized_datasets(self, datasets: DatasetDict, text_columns: Sequence[str] = ('sentence',), target_column: str = 'label') -> DatasetDict:
        def _tokenize_fn(batch: Dict[str, List[Union[str, int]]]) -> BatchEncoding:
            tokenized_inputs = self.tokenizer(
                *[batch[col] for col in text_columns],
                padding="max_length" if self.padding == 'max_length' else False,
                max_length=self.max_length,
                truncation=True,
            )
            if target_column in batch:
                tokenized_inputs['labels'] = batch[target_column]
            return tokenized_inputs
        tokenized_datasets = datasets.map(
            _tokenize_fn,
            batched=True,
            remove_columns=datasets[list(datasets.keys())[0]].column_names
        )
        return tokenized_datasets

    @property
    def data_collator(self) -> Any:
        if self._data_collator is None:
            self._data_collator = DataCollatorWithPadding(
                    tokenizer=self.tokenizer,
                    padding=self.padding,
                    max_length=self.max_length,
                    return_tensors='np')
        return self._data_collator

    def _collate(self, batch: Dict[str, Array], batch_size: int) -> Dict[str, Array]:
        if self.padding and self.padding != "max_length":
            batch = self.data_collator(batch)
        return batch

