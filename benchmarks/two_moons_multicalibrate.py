import flax.linen as nn
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from sklearn.datasets import make_moons

from fortuna.conformal import Multicalibrator
from fortuna.data import (
    DataLoader,
    InputsLoader,
)
from fortuna.metric.classification import accuracy
from fortuna.model.mlp import MLP
from fortuna.prob_model import (
    CalibConfig,
    CalibMonitor,
    FitConfig,
    FitMonitor,
    FitOptimizer,
    MAPPosteriorApproximator,
    ProbClassifier,
)

train_data = make_moons(n_samples=5000, noise=0.07, random_state=0)
val_data = make_moons(n_samples=1000, noise=0.07, random_state=1)
test_data = make_moons(n_samples=1000, noise=0.07, random_state=2)

train_data_loader = DataLoader.from_array_data(
    train_data, batch_size=128, shuffle=True, prefetch=True
)
val_data_loader = DataLoader.from_array_data(val_data, batch_size=128, prefetch=True)
test_data_loader = DataLoader.from_array_data(test_data, batch_size=128, prefetch=True)

output_dim = 2
prob_model = ProbClassifier(
    model=MLP(output_dim=output_dim, activations=(nn.tanh, nn.tanh)),
    posterior_approximator=MAPPosteriorApproximator(),
)

status = prob_model.train(
    train_data_loader=train_data_loader,
    val_data_loader=val_data_loader,
    calib_data_loader=val_data_loader,
    fit_config=FitConfig(
        monitor=FitMonitor(metrics=(accuracy,), early_stopping_patience=10),
        optimizer=FitOptimizer(method=optax.adam(1e-4), n_epochs=10),
    ),
    calib_config=CalibConfig(monitor=CalibMonitor(early_stopping_patience=2)),
)

test_inputs_loader = test_data_loader.to_inputs_loader()
test_means = prob_model.predictive.mean(inputs_loader=test_inputs_loader)
test_modes = prob_model.predictive.mode(
    inputs_loader=test_inputs_loader, means=test_means
)

fig = plt.figure(figsize=(6, 3))
size = 150
xx = np.linspace(-4, 4, size)
yy = np.linspace(-4, 4, size)
grid = np.array([[_xx, _yy] for _xx in xx for _yy in yy])
grid_loader = InputsLoader.from_array_inputs(grid)
grid_entropies = prob_model.predictive.entropy(grid_loader).reshape(size, size)
grid = grid.reshape(size, size, 2)
plt.title("Predictions and entropy", fontsize=12)
im = plt.pcolor(grid[:, :, 0], grid[:, :, 1], grid_entropies)
plt.scatter(
    test_data[0][:, 0],
    test_data[0][:, 1],
    s=1,
    c=["C0" if i == 1 else "C1" for i in test_modes],
)
plt.colorbar()
plt.show()

val_inputs_loader = val_data_loader.to_inputs_loader()
test_inputs_loader = test_data_loader.to_inputs_loader()
val_targets = val_data_loader.to_array_targets()
test_targets = test_data_loader.to_array_targets()

val_means = prob_model.predictive.mean(val_inputs_loader)
test_means = prob_model.predictive.mean(val_inputs_loader)

mc = Multicalibrator()
scores = val_targets
test_scores = test_targets
groups = jnp.stack((val_means.argmax(1) == 0, val_means.argmax(1) == 1), axis=1)
test_groups = jnp.stack((test_means.argmax(1) == 0, test_means.argmax(1) == 1), axis=1)
values = val_means[:, 1]
test_values = test_means[:, 1]
calib_test_values, status = mc.calibrate(
    scores=scores,
    groups=groups,
    values=values,
    test_groups=test_groups,
    test_values=test_values,
    n_buckets=1000,
)

plt.figure(figsize=(10, 3))
plt.suptitle("Multivalid calibration of probability that Y=1")
plt.subplot(1, 3, 1)
plt.title("all test inputs")
plt.hist([test_values, calib_test_values])[-1]
plt.legend(["before calibration", "after calibration"])
plt.xlabel("prob")
plt.subplot(1, 3, 2)
plt.title("inputs for which we predict 0")
plt.hist([test_values[test_groups[:, 0]], calib_test_values[test_groups[:, 0]]])[-1]
plt.xlabel("prob")
plt.subplot(1, 3, 3)
plt.title("inputs for which we predict 1")
plt.hist([test_values[test_groups[:, 1]], calib_test_values[test_groups[:, 1]]])[-1]
plt.xlabel("prob")
plt.tight_layout()
plt.show()

plt.title("Max calibration error decay during calibration")
plt.semilogy(status["max_calib_errors"])
plt.show()

print(
    "Per-group reweighted avg. squared calib. error before calibration: ",
    mc.calibration_error(
        scores=test_scores, groups=test_groups, values=test_means.max(1)
    ),
)
print(
    "Per-group reweighted avg. squared calib. error after calibration: ",
    mc.calibration_error(
        scores=test_scores, groups=test_groups, values=calib_test_values
    ),
)

print(
    "Mismatch between labels and probs before calibration: ",
    jnp.mean(
        jnp.maximum((1 - test_targets) * test_values, test_targets * (1 - test_values))
    ),
)
print(
    "Mismatch between labels and probs after calibration: ",
    jnp.mean(
        jnp.maximum(
            (1 - test_targets) * calib_test_values,
            test_targets * (1 - calib_test_values),
        )
    ),
)
