# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: nbsphinx
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: python3
#     language: python
#     name: python3
#   nbsphinx:
#     execute: never
# ---

# # Sentiment analysis

# In this notebook we show how to download a pre-trained model and a dataset from Hugging Face,
# and how to calibrate the model by fine-tuning part of its parameters for a sentiment analysis task.
#
# The following cell makes several configuration choices.
# By default, we use a [Bert](https://huggingface.co/docs/transformers/model_doc/bert#transformers.FlaxBertModel) model
# and the [imdb](https://huggingface.co/datasets/imdb) dataset. We make choices on how to split the data,
# on the batch size and on the optimization.

# +
pretrained_model_name_or_path = "bert-base-cased"
dataset_name = "imdb"
text_columns = ("text",)
target_column = "label"
train_split = "train[:2%]+train[-2%:]"
val_split = "test[:1%]+test[-1%:]"
test_split = "test[:1%]+test[-1%:]"
num_labels = 2
max_sequence_length = 512
per_device_train_batch_size = 16
per_device_eval_batch_size = 16
seed = 0

weight_decay = 0.01
num_train_epochs = 2
num_warmup_steps = 0
learning_rate = 2e-5
max_grad_norm = 1.0
early_stopping_patience = 1
# -

# ## Prepare the data

# First thing first, from Hugging Face we instantiate a tokenizer for the pre-trained model in use.

# +
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
# -

# Then, we download some calibration, validation and test datasets.

# +
from datasets import DatasetDict, load_dataset

datasets = DatasetDict(
    {
        "calibration": load_dataset(dataset_name, split=train_split),
        "validation": load_dataset(dataset_name, split=val_split),
        "test": load_dataset(dataset_name, split=test_split),
    }
)
# -

# It's time for Fortuna to come into play. First, we call a sequence classification dataset object,
# then we tokenize the datasets, and finally we construct calibration, validation, and test data loaders.

# +
from fortuna.data.dataset.huggingface_datasets import (
    HuggingFaceSequenceClassificationDataset,
)
import jax

hf_dataset = HuggingFaceSequenceClassificationDataset(
    tokenizer=tokenizer,
    padding="max_length",
    max_length=max_sequence_length,
    num_unique_labels=num_labels,
)

datasets = hf_dataset.get_tokenized_datasets(
    datasets, text_columns=text_columns, target_column=target_column
)

rng = jax.random.PRNGKey(seed)

calib_data_loader = hf_dataset.get_data_loader(
    datasets["calibration"],
    per_device_batch_size=per_device_train_batch_size,
    shuffle=True,
    drop_last=True,
    rng=rng,
)
val_data_loader = hf_dataset.get_data_loader(
    datasets["validation"],
    per_device_batch_size=per_device_eval_batch_size,
    shuffle=False,
    drop_last=False,
    rng=rng,
)
test_data_loader = hf_dataset.get_data_loader(
    datasets["test"],
    per_device_batch_size=per_device_eval_batch_size,
    shuffle=False,
    drop_last=False,
    rng=rng,
    verbose=True,
)
# -

# ## Define the transformer model

# From the [transformers](https://huggingface.co/docs/transformers/index) library of Hugging Face,
# we instantiate the pre-trained transformer of interest.

# +
from transformers import FlaxAutoModelForSequenceClassification

model = FlaxAutoModelForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path=pretrained_model_name_or_path, num_labels=num_labels
)
# -

# We now pass the model to `CalibClassifier`, and instantiate a calibration model. Out-of-the-box, you will be able to
# finetune your model on custom loss functions, and on arbitrary subsets of model parameters.

# +
from fortuna.calib_model import CalibClassifier

calib_model = CalibClassifier(model=model)
# -

# ## Calibrate!

# We first construct an optimizer. We use Fortuna's functionality to define a learning rate scheduler for
# [AdamW](https://arxiv.org/pdf/1711.05101.pdf).

# +
from fortuna.utils.optimizer import (
    linear_scheduler_with_warmup,
    decay_mask_without_layer_norm_fn,
)
import optax

optimizer = optax.adamw(
    learning_rate=linear_scheduler_with_warmup(
        learning_rate=learning_rate,
        num_inputs_train=len(datasets["calibration"]),
        train_total_batch_size=per_device_train_batch_size * jax.local_device_count(),
        num_train_epochs=num_train_epochs,
        num_warmup_steps=num_warmup_steps,
    ),
    weight_decay=weight_decay,
    mask=decay_mask_without_layer_norm_fn,
)
# -

# We then configure the calibration process, in particular hyperparameters, metrics to monitor, early stopping,
# the optimizer and which parameters we want to calibrate. Here, we are choosing to calibrate only the parameters that
# contain "classifier" in the path, i.e. only the parameters of the last layer.

# +
from fortuna.calib_model import Config, Optimizer, Monitor, Hyperparameters
from fortuna.metric.classification import accuracy, brier_score


def acc(preds, uncertainties, targets):
    return accuracy(preds, targets)


def brier(preds, uncertainties, targets):
    return brier_score(uncertainties, targets)


config = Config(
    hyperparameters=Hyperparameters(
        max_grad_norm=max_grad_norm,
    ),
    monitor=Monitor(
        metrics=(acc, brier),
        early_stopping_patience=1,
    ),
    optimizer=Optimizer(
        method=optimizer,
        n_epochs=num_train_epochs,
        freeze_fun=lambda path, v: "trainable" if "classifier" in path else "frozen",
    ),
)
# -

# Finally, we calibrate! By default, the method employs a
# [focal loss](https://proceedings.neurips.cc/paper/2020/file/aeb7b30ef1d024a76f21a1d40e30c302-Paper.pdf),
# but feel free to pass your favourite one!

status = calib_model.calibrate(
    calib_data_loader=calib_data_loader, val_data_loader=val_data_loader, config=config
)

# ## Compute metrics

# We now compute some accuracy and [Expected Calibration Error](http://proceedings.mlr.press/v70/guo17a/guo17a.pdf)
# (ECE) to evaluate how the method performs on some test data.

# +
from fortuna.metric.classification import expected_calibration_error

test_inputs_loader = test_data_loader.to_inputs_loader()
means = calib_model.predictive.mean(
    inputs_loader=test_inputs_loader,
)
modes = calib_model.predictive.mode(
    inputs_loader=test_inputs_loader,
)

test_targets = test_data_loader.to_array_targets()
acc = accuracy(preds=modes, targets=test_targets)
ece = expected_calibration_error(
    preds=modes,
    probs=means,
    targets=test_targets,
)
# -

print(f"Accuracy on test set: {acc}.")
print(f"ECE on test set: {ece}.")
