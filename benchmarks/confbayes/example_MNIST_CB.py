import flax.linen as nn
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds

from fortuna.conformal import AdaptivePredictionConformalClassifier
from fortuna.data import (
    DataLoader,
    InputsLoader,
)
from fortuna.metric.classification import (
    accuracy,
    brier_score,
    expected_calibration_error,
)
from fortuna.model import (
    MLP,
    LeNet5,
    cnn,
)
from fortuna.prob_model import (
    ADVIPosteriorApproximator,
    CalibConfig,
    CalibMonitor,
    FitConfig,
    FitMonitor,
    FitOptimizer,
    ProbClassifier,
)


# Approximate Conformal Bayes: MNIST Example
## Download data
def download(split_range, shuffle=False):
    ds = tfds.load(
        name="MNIST",
        split=f"train[{split_range}]",
        as_supervised=True,
        shuffle_files=True,
    ).map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
    if shuffle:
        ds = ds.shuffle(10, reshuffle_each_iteration=True)
    return ds.batch(128).prefetch(1)


train_data_loader, val_data_loader, test_data_loader = (
    download(":10%", shuffle=True),
    download("10%:20%"),
    download("20%:30%"),
)

## Initialize constants
n_posterior_samples = 100

## Fit Flax models
# Setup DataLoaders
train_data_loader = DataLoader.from_tensorflow_data_loader(train_data_loader)
test_data_loader = DataLoader.from_tensorflow_data_loader(test_data_loader)
test_inputs_loader = test_data_loader.to_inputs_loader()
test_data = test_data_loader.to_array_data()
val_data_loader = DataLoader.from_tensorflow_data_loader(val_data_loader)


# # Setup Flax Model
output_dim = 10
prob_model = ProbClassifier(
    model=LeNet5(output_dim=output_dim),
    posterior_approximator=ADVIPosteriorApproximator(),
)

# Train model
status = prob_model.train(
    train_data_loader=train_data_loader,
    fit_config=FitConfig(
        optimizer=FitOptimizer(
            freeze_fun=lambda path, val: "trainable"
            if "output_subnet" in path
            else "frozen"
        )
    ),
    map_fit_config=FitConfig(
        monitor=FitMonitor(early_stopping_patience=2, metrics=(accuracy,)),
        optimizer=FitOptimizer(),
    ),
)


### Run Conformal Bayes
error = 0.1
conformal_set, ESS = prob_model.predictive.conformal_set(
    train_data_loader,
    test_inputs_loader,
    error=error,
    n_posterior_samples=n_posterior_samples,
    return_ess=True,
)  # Note that we do not need val_data, so we could merge it with train_data

### Evaluate conformal Bayes
# Evaluate conformal method
y_test = test_data[1]  # test data points
n_test = jnp.shape(y_test)[0]

# Loop through test data points and compute coverage and length of conformal sets
coverage = jnp.array([y_test[j] in conformal_set[j] for j in range(n_test)])
length = jnp.array([len(conformal_set[j]) for j in range(n_test)])

print("Coverage for CB in test set = {}".format(jnp.mean(coverage)))
print("Mean set length for CB = {}".format(jnp.mean(length)))
print("Mean effective sample size for IS = {}".format(jnp.mean(ESS)))

### Bayesian credible sets
# Naive Bayes predictive method and evaluation
credible_set = prob_model.predictive.credible_set(
    test_inputs_loader, n_posterior_samples=n_posterior_samples, error=error
)

# Loop through test data points and compute coverage and length of credible sets
coverage_bayes = jnp.array([y_test[j] in credible_set[j] for j in range(n_test)])
length_bayes = jnp.array([len(credible_set[j]) for j in range(n_test)])

print("Coverage for Bayes in test set = {}".format(jnp.mean(coverage_bayes)))
print("Mean set length for Bayes = {}".format(jnp.mean(length_bayes)))

### Adaptive Prediction Conformal Classifier Sets
val_means = prob_model.predictive.mean(
    inputs_loader=val_data_loader.to_inputs_loader(),
    n_posterior_samples=n_posterior_samples,
)
test_means = prob_model.predictive.mean(
    inputs_loader=test_inputs_loader, n_posterior_samples=n_posterior_samples
)
conformal_set_aps = AdaptivePredictionConformalClassifier().conformal_set(
    val_probs=val_means,
    test_probs=test_means,
    val_targets=val_data_loader.to_array_targets(),
    error=error,
)

# Loop through test data points and compute coverage and length of conformal sets
coverage = jnp.array([y_test[j] in conformal_set_aps[j] for j in range(n_test)])
length = jnp.array([len(conformal_set_aps[j]) for j in range(n_test)])

print("Coverage for APS in test set = {}".format(jnp.mean(coverage)))
print("Mean set length for APS = {}".format(jnp.mean(length)))
