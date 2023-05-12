from collections import OrderedDict

from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

from fortuna.prob_model.posterior.sngp.transformers.auto_factory import (
    _BaseAutoSNGPModelClass,
    _SNGPLazyAutoMapping,
)

FLAX_SNGP_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Sequence Classification with SNGP mapping
        ("bert", "FlaxSNGPBertExtractorForSequenceClassification"),
        ("distilbert", "FlaxSNGPDistilBertExtractorForSequenceClassification"),
        ("roberta", "FlaxSNGPRobertaExtractorForSequenceClassification"),
    ]
)

FLAX_SNGP_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING = _SNGPLazyAutoMapping(
    CONFIG_MAPPING_NAMES, FLAX_SNGP_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES
)


class FlaxAutoSNGPModelForSequenceClassification(_BaseAutoSNGPModelClass):
    _model_mapping = FLAX_SNGP_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING
