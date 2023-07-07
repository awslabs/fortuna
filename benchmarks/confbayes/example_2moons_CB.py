import flax.linen as nn
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpy as np
import optax
from sklearn.datasets import make_moons

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

## Fit Flax models
# Setup DataLoaders
train_data_loader = DataLoader.from_array_data(
    train_data, batch_size=128, shuffle=True, prefetch=True
)

test_inputs_loader = InputsLoader.from_array_inputs(
    test_data[0], batch_size=128, prefetch=True
)

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

### Run conformal Bayes
error = 0.2
conformal_set, ESS = prob_model.predictive.conformal_set(
    train_data_loader,
    test_inputs_loader,
    error=error,
    n_posterior_samples=100,
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


### Evaluate regular Bayesian credible sets
# Naive Bayes predictive method and evaluation
p_bayes = prob_model.predictive.mean(inputs_loader=test_inputs_loader)[
    :, 1
]  # P(Y =1 |x) under Bayesian model

credible_set_bayes = np.zeros((n_test, 2))
coverage_bayes = np.zeros(n_test)
length_bayes = np.zeros(n_test)

for j in range(n_test):
    # Compute region from p_bayes
    if p_bayes[j] > (1 - error):  # only y = 1
        credible_set_bayes[j] = np.array([0, 1])
    elif (1 - p_bayes[j]) > (1 - error):  # only y = 0
        credible_set_bayes[j] = np.array([1, 0])
    else:
        credible_set_bayes[j] = np.array([1, 1])

    coverage_bayes[j] = credible_set_bayes[j][y_test[j]]
    length_bayes[j] = jnp.sum(credible_set_bayes[j])

print("Coverage for Bayes in test set = {}".format(jnp.mean(coverage_bayes)))
print("Mean set length for Bayes = {}".format(jnp.mean(length_bayes)))
