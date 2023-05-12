import abc
import logging
from typing import Union, Dict, List, Iterable, Optional, Tuple, Sequence, Any

import jax.random
from datasets import DatasetDict, Dataset
from jax import numpy as jnp
from jax.random import PRNGKeyArray
from tqdm import tqdm
from transformers import (
    PreTrainedTokenizer,
    BatchEncoding,
    DataCollatorWithPadding,
)
from transformers.utils import PaddingStrategy

from fortuna.data.dataset.data_collator import (
    FlaxDataCollatorForMultipleChoice,
    FlaxDataCollatorForLanguageModeling,
)
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
        pad_to_multiple_of: int = 64,
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
        pad_to_multiple_of: int
            Pad the sequence to a multiple of the provided value.
            This argument will be used whenever padding is not done to a fixed constant (e.g., max_length).
        num_unique_labels: Optional[int]
            Number of unique target labels in the task (classification only)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.pad_to_multiple_of = pad_to_multiple_of
        self.num_unique_labels = num_unique_labels
        self._data_collator = None

    @abc.abstractmethod
    def get_tokenized_datasets(
        self, datasets: DatasetDict, *args, **kwargs
    ) -> DatasetDict:
        pass

    @property
    @abc.abstractmethod
    def data_collator(self):
        """
        Create a data collator object that will process batches of data and apply transformation on them if needed.
        """
        pass

    def get_data_loader(
        self,
        dataset: Dataset,
        per_device_batch_size: int,
        rng: PRNGKeyArray,
        shuffle: bool = False,
        drop_last: bool = False,
        verbose: bool = False,
    ) -> HuggingFaceDataLoader:
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
            num_unique_labels=self.num_unique_labels,
        )

    @abc.abstractmethod
    def _collate(self, batch: Dict[str, Array], batch_size: int) -> Dict[str, Array]:
        pass

    @staticmethod
    def _get_batches_idxs(
        rng: PRNGKeyArray,
        dataset_size: int,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> Iterable[Array]:
        if shuffle:
            dataset_idxs = jax.random.permutation(rng, dataset_size)  # batch idxs
        else:
            dataset_idxs = jnp.arange(dataset_size)
        if drop_last:
            steps_per_epoch = dataset_size // batch_size
            dataset_idxs = dataset_idxs[
                : steps_per_epoch * batch_size
            ]  # Skip incomplete batch.
        yield from [
            dataset_idxs[start : start + batch_size]
            for start in range(0, len(dataset_idxs), batch_size)
        ]

    def _get_data_loader(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        rng: Optional[PRNGKeyArray] = None,
        verbose: bool = False,
    ) -> Union[Iterable[Dict[str, Array]], Iterable[Tuple[Dict[str, Array], Array]]]:
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
            batch_idxs_gen = tqdm(
                batch_idxs_gen, desc="Batches:", total=steps_per_epoch
            )
        for batch_idxs in batch_idxs_gen:
            batch = dataset[batch_idxs]
            batch = self._collate(batch, len(batch["input_ids"]))
            batch_inputs = {k: jnp.array(v) for k, v in batch.items()}
            if "labels" in batch_inputs:
                batch_labels = jnp.array(batch_inputs.pop("labels"))
                yield (batch_inputs, batch_labels)
            else:
                yield batch_inputs, None


class HuggingFaceSequenceClassificationDataset(HuggingFaceClassificationDatasetABC):
    """
    Dataset for Sequence Classification tasks.
    """

    def get_tokenized_datasets(
        self,
        datasets: DatasetDict,
        text_columns: Sequence[str] = ("sentence",),
        target_column: str = "label",
        **kwargs,
    ) -> DatasetDict:
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

        def _tokenize_fn(batch: Dict[str, List[Union[str, int]]]) -> BatchEncoding:
            tokenized_inputs = self.tokenizer(
                *[batch[col] for col in text_columns],
                padding="max_length" if self.padding == "max_length" else False,
                max_length=self.max_length,
                truncation=True,
            )
            if target_column in batch:
                tokenized_inputs["label"] = batch[target_column]
            return tokenized_inputs

        tokenized_datasets = datasets.map(
            _tokenize_fn,
            batched=True,
            remove_columns=datasets[list(datasets.keys())[0]].column_names,
        )
        return tokenized_datasets

    @property
    def data_collator(self) -> Any:
        if self._data_collator is None:
            self._data_collator = DataCollatorWithPadding(
                tokenizer=self.tokenizer,
                padding=self.padding,
                max_length=self.max_length,
                return_tensors="np",
            )
        return self._data_collator

    def _collate(self, batch: Dict[str, Array], batch_size: int) -> Dict[str, Array]:
        if self.padding and self.padding != "max_length":
            batch = self.data_collator(batch)
        else:
            if "label" in batch:
                batch["labels"] = batch["label"]
                del batch["label"]
        return batch


class HuggingFaceMultiChoiceDataset(HuggingFaceClassificationDatasetABC):
    """
    Dataset for Multi Choice classification tasks.
    """

    def get_tokenized_datasets(
        self,
        datasets: DatasetDict,
        contexts: Sequence[str] = (),
        choices: Sequence[str] = (),
        target_column: str = "label",
        **kwargs,
    ) -> DatasetDict:
        """
        Converts one or more dataset into sequences of tokens, using the tokenizer.

        Parameters
        ----------
        datasets: DatasetDict
            A dictionary of `datasets.Dataset` that have to be encoded.
        contexts: Sequence[str]
            A list of str containing the column names for the contexts.
            The first column contain the initial context (the first sentence).
            The second column name contains the beginning of the second sentence.
        choices:
            A list of str containing the column names for the possible continuations of the context.
        target_column: str
            The target column name
        Returns
        -------
        DatasetDict
            A dictionary of tokenized datasets.
        """
        sent1 = contexts[0]
        sent2 = contexts[1]

        def _tokenize_fn(batch: Dict[str, List[Union[str, int]]]) -> Dict:
            first_sentences = [
                [context] * self.num_unique_labels for context in batch[sent1]
            ]
            question_headers = batch[sent2]
            if len(choices) > 1:
                second_sentences = [
                    [f"{header} {batch[end][i]}" for end in choices]
                    for i, header in enumerate(question_headers)
                ]
            else:
                second_sentences = [
                    [f"{header} {end}" for end in batch[choices[0]][i]]
                    for i, header in enumerate(question_headers)
                ]
            first_sentences = sum(first_sentences, [])
            second_sentences = sum(second_sentences, [])
            tokenized_inputs = self.tokenizer(
                first_sentences, second_sentences, truncation=True
            )
            tokenized_inputs = {
                k: [
                    v[i : i + self.num_unique_labels]
                    for i in range(0, len(v), self.num_unique_labels)
                ]
                for k, v in tokenized_inputs.items()
            }
            tokenized_inputs["label"] = batch[target_column]
            return tokenized_inputs

        tokenized_datasets = datasets.map(
            _tokenize_fn,
            batched=True,
            remove_columns=datasets[list(datasets.keys())[0]].column_names,
        )
        return tokenized_datasets

    @property
    def data_collator(self) -> Any:
        if self._data_collator is None:
            self._data_collator = FlaxDataCollatorForMultipleChoice(
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
        return self._data_collator

    def _collate(self, batch: Dict[str, Array], batch_size: int) -> Dict[str, Array]:
        batch = [{k: batch[k][i] for k in batch.keys()} for i in range(batch_size)]
        batch = self.data_collator(batch)
        return batch


class HuggingFaceMaskedLMDataset(HuggingFaceClassificationDatasetABC):
    def __init__(
        self,
        *args,
        mlm_probability: Optional[float] = 0.15,
        **kwargs,
    ):
        """
        Parameters
        ----------
        mlm_probability Optional[float]
            The probability with which to (randomly) mask tokens in the input,
            when the task is masked language modeling.
        """
        super(HuggingFaceMaskedLMDataset, self).__init__(*args, **kwargs)
        if not self.tokenizer.is_fast:
            logger.warning(
                f"You are not using a Fast Tokenizer, so whole words cannot be masked, only tokens."
            )
        self.mlm_probability = mlm_probability

    def get_tokenized_datasets(
        self,
        datasets: DatasetDict,
        text_columns: Sequence[str] = ("sentence",),
        **kwargs,
    ) -> DatasetDict:
        """
        Converts one or more dataset into sequences of tokens, using the tokenizer.

        Parameters
        ----------
        datasets: DatasetDict
            A dictionary of `datasets.Dataset` that have to be encoded.
        text_columns: str
            A list containing the text column names, whose text sequences has to be encoded.

        Returns
        -------
        DatasetDict
            A dictionary of tokenized datasets.
        """
        assert (
            len(text_columns) == 1
        ), "Only one text column should be passed when the task is MaskedLM."

        def _tokenize_fn(
            batch: Dict[str, List[Union[str, int]]]
        ) -> Dict[str, List[int]]:
            tokenized_inputs = self.tokenizer(
                *[batch[col] for col in text_columns],
                truncation=False,
                return_special_tokens_mask=True,
            )
            # Concatenate all texts
            keys = list(tokenized_inputs.keys())
            tokenized_inputs_cat = {k: sum(tokenized_inputs[k], []) for k in keys}
            # Compute length of concatenated texts
            total_length = len(tokenized_inputs_cat[keys[0]])
            # We drop the last chunk if it's smaller than chunk_size
            total_length = (total_length // self.max_length) * self.max_length
            # Split by chunks of max_len
            tokenized_inputs = {
                k: [
                    t[i : i + self.max_length]
                    for i in range(0, total_length, self.max_length)
                ]
                for k, t in tokenized_inputs_cat.items()
            }
            return tokenized_inputs

        tokenized_datasets = datasets.map(
            _tokenize_fn,
            batched=True,
            remove_columns=datasets[list(datasets.keys())[0]].column_names,
        )
        return tokenized_datasets

    @property
    def data_collator(self):
        if self._data_collator is None:
            self._data_collator = FlaxDataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm_probability=self.mlm_probability,
            )
        return self._data_collator

    def _collate(self, batch: Dict[str, Array], batch_size: int) -> Dict[str, Array]:
        batch = [{k: batch[k][i] for k in batch.keys()} for i in range(batch_size)]
        return self.data_collator(batch, pad_to_multiple_of=self.pad_to_multiple_of)
