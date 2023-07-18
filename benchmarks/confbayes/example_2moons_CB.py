import flax.linen as nn
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpy as np
import optax
from sklearn.datasets import make_moons

from fortuna.conformal import AdaptivePredictionConformalClassifier
from fortuna.data import (
    DataLoader,
    InputsLoader,
)
from fortuna.metric.classification import accuracy
from fortuna.model import MLP
from fortuna.prob_model import (
    ADVIPosteriorApproximator,
    CalibConfig,
    CalibMonitor,
    FitConfig,
    FitMonitor,
    FitOptimizer,
    ProbClassifier,
)

# Approximate Conformal Bayes: 2 Moons Example
## Simulate data
n_train = 1000
train_data = make_moons(n_samples=n_train, noise=0.2, random_state=0)
n_test = 500
test_data = make_moons(n_samples=n_test, noise=0.2, random_state=1)
n_val = 500
val_data = make_moons(n_samples=n_val, noise=0.2, random_state=2)

## Initialize constants
n_posterior_samples = 100

## Fit Flax models
# Setup DataLoaders
train_data_loader = DataLoader.from_array_data(
    train_data, batch_size=128, shuffle=True, prefetch=True
)

test_inputs_loader = InputsLoader.from_array_inputs(
    test_data[0], batch_size=128, prefetch=True
)

val_data_loader = DataLoader.from_array_data(val_data, batch_size=128, prefetch=True)


# Setup Flax Model
output_dim = 2
prob_model = ProbClassifier(
    model=MLP(output_dim=output_dim, activations=(nn.tanh, nn.tanh)),
    posterior_approximator=ADVIPosteriorApproximator(),
)

# Train model
status = prob_model.train(
    train_data_loader=train_data_loader,  # Remove validation set to make it simpler
    fit_config=FitConfig(
        monitor=FitMonitor(metrics=(accuracy,), early_stopping_patience=10),
        optimizer=FitOptimizer(method=optax.adam(1e-1)),
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
)


### Evaluate conformal Bayes
# Evaluate conformal method
y_test = test_data[1]  # test data points

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
