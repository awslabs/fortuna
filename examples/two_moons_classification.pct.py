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
# ---

# %% [markdown]
# # Two-moons Classification

# %% [markdown]
# In this notebook we show how to use Fortuna to obtain calibrated uncertainty estimates of predictions in an MNIST classification task.

# %% [markdown]
# ### Download Two-Moons data from scikit-learn
# Let us first download two-moons data from [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html).

# %%
from sklearn.datasets import make_moons

train_data = make_moons(n_samples=500, noise=0.07, random_state=0)
val_data = make_moons(n_samples=500, noise=0.07, random_state=1)
test_data = make_moons(n_samples=500, noise=0.07, random_state=2)

# %% [markdown]
# ### Convert data to a compatible data loader
# Fortuna helps you converting data and data loaders into a data loader that Fortuna can digest.

# %%
from fortuna.data import DataLoader

train_data_loader = DataLoader.from_array_data(
    train_data, batch_size=128, shuffle=True, prefetch=True
)
val_data_loader = DataLoader.from_array_data(val_data, batch_size=128, prefetch=True)
test_data_loader = DataLoader.from_array_data(test_data, batch_size=128, prefetch=True)

# %% [markdown]
# ### Build a probabilistic classifier
# Let us build a probabilistic classifier. This is an interface object containing several attributes that you can configure, i.e. `model`, `prior`, `posterior_approximator`, `output_calibrator`. In this example, we use an MLP model, an Automatic Differentiation Variational Inference posterior approximator, and the default temperature scaling output calibrator.

# %%
from fortuna.prob_model import ProbClassifier, ADVIPosteriorApproximator
from fortuna.model import MLP
import flax.linen as nn

output_dim = 2
prob_model = ProbClassifier(
    model=MLP(output_dim=output_dim, activations=(nn.tanh, nn.tanh)),
    posterior_approximator=ADVIPosteriorApproximator(),
)

# %% [markdown]
# ### Train the probabilistic model: posterior fitting and calibration
# We can now train the probabilistic model. This includes fitting the posterior distribution and calibrating the probabilistic model.

# %%
from fortuna.prob_model import FitConfig, FitMonitor, FitOptimizer, CalibConfig, CalibMonitor
from fortuna.metric.classification import accuracy
import optax

status = prob_model.train(
    train_data_loader=train_data_loader,
    val_data_loader=val_data_loader,
    calib_data_loader=val_data_loader,
    fit_config=FitConfig(
        monitor=FitMonitor(metrics=(accuracy,), early_stopping_patience=2),
        optimizer=FitOptimizer(method=optax.adam(1e-1)),
    ),
    calib_config=CalibConfig(monitor=CalibMonitor(early_stopping_patience=2))
)

# %% [markdown]
# ### Estimate predictive statistics
# We can now compute some predictive statistics by invoking the `predictive` attribute of the probabilistic classifier, and the method of interest. Most predictive statistics, e.g. mean or mode, require a loader of input data points. You can easily get this from the data loader calling its method `to_inputs_loader`.

# %%
test_log_probs = prob_model.predictive.log_prob(data_loader=test_data_loader)
test_inputs_loader = test_data_loader.to_inputs_loader()
test_means = prob_model.predictive.mean(inputs_loader=test_inputs_loader)
test_modes = prob_model.predictive.mode(
    inputs_loader=test_inputs_loader, means=test_means
)

# %%
import matplotlib.pyplot as plt
from fortuna.data import InputsLoader
import numpy as np

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

# %% [markdown]
# ### Compute metrics
# In classification, the predictive mode is a prediction for labels, while the predictive mean is a prediction for the probability of each label. As such, we can use these to compute several metrics, e.g. the accuracy, the Brier score, the expected calibration error (ECE), etc.

# %%
from fortuna.metric.classification import (
    accuracy,
    expected_calibration_error,
    brier_score,
)

test_targets = test_data_loader.to_array_targets()
acc = accuracy(preds=test_modes, targets=test_targets)
brier = brier_score(probs=test_means, targets=test_targets)
ece = expected_calibration_error(
    preds=test_modes,
    probs=test_means,
    targets=test_targets,
    plot=True,
    plot_options=dict(figsize=(10, 2)),
)
print(f"Test accuracy: {acc}")
print(f"Brier score: {brier}")
print(f"ECE: {ece}")

# %% [markdown] pycharm={"name": "#%% md\n"}
# ### What if we have model outputs to start from?

# %% [markdown]
# If you have already trained a model and obtained model outputs, you can still use Fortuna to calibrate them, and estimate uncertainty. For educational purposes only, let us take the logarithm of the predictive mean estimated above as model outputs, and pretend these were generated with some other framework. Furthermore, we store arrays of validation and test target variables, and assume these were also given.

# %% pycharm={"name": "#%%\n"}
import numpy as np

calib_outputs = np.log(
    1e-6 + prob_model.predictive.mean(inputs_loader=val_data_loader.to_inputs_loader())
)
test_outputs = np.log(1e-6 + test_means)

calib_targets = val_data_loader.to_array_targets()
test_targets = test_data_loader.to_array_targets()

# %% [markdown]
# We now invoke a calibration classifier, with default temperature scaling output calibrator, and calibrate the model outputs.

# %% pycharm={"name": "#%%\n"}
from fortuna.output_calib_model import OutputCalibClassifier, Config, Monitor

calib_model = OutputCalibClassifier()
calib_status = calib_model.calibrate(
    calib_outputs=calib_outputs, calib_targets=calib_targets,
    config=Config(monitor=Monitor(early_stopping_patience=2))
)

# %% [markdown]
# Similarly as above, we can now compute predictive statistics.

# %% pycharm={"name": "#%%\n"}
test_log_probs = calib_model.predictive.log_prob(
    outputs=test_outputs, targets=test_targets
)
test_means = calib_model.predictive.mean(outputs=test_outputs)
test_modes = calib_model.predictive.mode(outputs=test_outputs)

# %% [markdown] pycharm={"name": "#%% md\n"}
# Then one can compute metrics, exactly as done above.
