# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: python3
#     language: python
#     name: python3
#   nbsphinx:
#     execute: never
# ---

# %%
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

# %%
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

# %%
from datasets import DatasetDict, load_dataset

datasets = DatasetDict(
    {
        "calibration": load_dataset(dataset_name, split=train_split),
        "validation": load_dataset(dataset_name, split=val_split),
        "test": load_dataset(dataset_name, split=test_split),
    }
)

# %%
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

# %%
from transformers import FlaxAutoModelForSequenceClassification

model = FlaxAutoModelForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path=pretrained_model_name_or_path, num_labels=num_labels
)

# %%
from fortuna.calib_model import CalibClassifier

calib_model = CalibClassifier(model=model)

# %%
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

# %%
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

# %%
status = calib_model.calibrate(
    calib_data_loader=calib_data_loader, val_data_loader=val_data_loader, config=config
)

# %%
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

# %%
print(f"Accuracy on test set: {acc}.")
print(f"ECE on test set: {ece}.")
