from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import flax
from jax import numpy as jnp
import numpy as np
from transformers import (
    PreTrainedTokenizerBase,
    TensorType,
)
from transformers.utils import PaddingStrategy


@flax.struct.dataclass
class FlaxDataCollatorForMultipleChoice:
    """
    Data collator used for multiple choice tasks like SWAG.
    Inputs are dynamically padded to either the maximum length of a batch or the provide `max_length`.

    Attributes
    ----------
    tokenizer: Union[:class:`~transformers.PreTrainedTokenizer`, :class:`~transformers.PreTrainedTokenizerFast`]
        The tokenizer used for encoding the data.
    padding: Union[bool, str, `~utils.PaddingStrategy`]:
        Select a strategy to pad the returned sequences (according to the model's padding side and padding
        index) among:

        - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
        sequence if provided).
        - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
        acceptable input length for the model if that argument is not provided.
        - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different
        lengths).
    max_length: Optional[int]
        Maximum length of the returned list and optionally padding length (see above).
    pad_to_multiple_of: Optional[int]
        If set will pad the sequence to a multiple of the provided value.

    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)]
            for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="np",
        )

        batch = {
            k: jnp.reshape(v, (batch_size, num_choices, -1)) for k, v in batch.items()
        }
        batch["labels"] = jnp.array(labels, dtype=jnp.int32)
        return batch


@flax.struct.dataclass
class FlaxDataCollatorForLanguageModeling:
    # Obtained from from https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_mlm_flax.py
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.
    Attributes
    ----------
    tokenizer: Union[:class:`~transformers.PreTrainedTokenizer`, :class:`~transformers.PreTrainedTokenizerFast`]
        The tokenizer used for encoding the data.
    mlm_probability: float
        The probability with which to (randomly) mask tokens in the input.
    .. note::
        For best performance, this data collator should be used with a dataset having items that are dictionaries or
        BatchEncoding, with the :obj:`"special_tokens_mask"` key, as returned by a
        :class:`~transformers.PreTrainedTokenizer` or a :class:`~transformers.PreTrainedTokenizerFast` with the
        argument :obj:`return_special_tokens_mask=True`.
    """

    tokenizer: PreTrainedTokenizerBase
    mlm_probability: float = 0.15

    def __post_init__(self):
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

    def __call__(
        self, examples: List[Dict[str, np.ndarray]], pad_to_multiple_of: int
    ) -> Dict[str, np.ndarray]:
        # Handle dict or lists with proper padding and conversion to tensor.
        batch = self.tokenizer.pad(
            examples,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=TensorType.NUMPY,
        )

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)

        batch["input_ids"], batch["labels"] = self.mask_tokens(
            batch["input_ids"], special_tokens_mask=special_tokens_mask
        )
        return batch

    def mask_tokens(
        self, inputs: np.ndarray, special_tokens_mask: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.copy()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = np.full(labels.shape, self.mlm_probability)
        special_tokens_mask = special_tokens_mask.astype("bool")

        probability_matrix[special_tokens_mask] = 0.0
        masked_indices = np.random.binomial(1, probability_matrix).astype("bool")
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            np.random.binomial(1, np.full(labels.shape, 0.8)).astype("bool")
            & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # 10% of the time, we replace masked input tokens with random word
        indices_random = np.random.binomial(1, np.full(labels.shape, 0.5)).astype(
            "bool"
        )
        indices_random &= masked_indices & ~indices_replaced

        random_words = np.random.randint(
            self.tokenizer.vocab_size, size=labels.shape, dtype="i4"
        )
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
