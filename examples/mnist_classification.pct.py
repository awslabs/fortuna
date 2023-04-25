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
# # MNIST Classification

# %% [markdown]
# In this notebook we show how to use Fortuna to obtain calibrated uncertainty estimates of predictions in an MNIST classification task, starting from scratch. In the last section of this example shows how this could have been done starting directly from outputs of a pre-trained model.

# %% [markdown]
# ### Download MNIST data from TensorFlow
# Let us first download the MNIST data from [TensorFlow Datasets](https://www.tensorflow.org/datasets). Other sources would be equivalently fine.

# %%
import tensorflow as tf
import tensorflow_datasets as tfds


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
    download(":80%", shuffle=True),
    download("80%:90%"),
    download("90%:"),
)

# %% [markdown]
# ### Convert data to a compatible data loader
# Fortuna helps you converting data and data loaders into a data loader that Fortuna can digest.

# %%
from fortuna.data import DataLoader

train_data_loader = DataLoader.from_tensorflow_data_loader(train_data_loader)
val_data_loader = DataLoader.from_tensorflow_data_loader(val_data_loader)
test_data_loader = DataLoader.from_tensorflow_data_loader(test_data_loader)

# %% [markdown]
# ### Build a probabilistic classifier
# Let us build a probabilistic classifier. This is an interface object containing several attributes that you can configure, i.e. `model`, `prior`, `posterior_approximator`, `output_calibrator`. In this example, we use a LeNet5 model, a Laplace posterior approximator acting on the last layer on the model, and the default temperature scaling output calibrator.

# %%
from fortuna.prob_model import ProbClassifier, LaplacePosteriorApproximator
from fortuna.model import LeNet5

output_dim = 10
prob_model = ProbClassifier(
    model=LeNet5(output_dim=output_dim),
    posterior_approximator=LaplacePosteriorApproximator(
        which_params=(["model", "params", "output_subnet"],)
    ),
)


# %% [markdown]
# ### Train the probabilistic model: posterior fitting and calibration
# We can now train the probabilistic model. This includes fitting the posterior distribution and calibrating the probabilistic model. As we are using a Laplace approximation, which start from a Maximum-A-Posteriori (MAP) approximation, we configure MAP via the argument `map_fit_config`.

# %%
from fortuna.prob_model import FitConfig, FitMonitor
from fortuna.metric.classification import accuracy

status = prob_model.train(
    train_data_loader=train_data_loader,
    val_data_loader=val_data_loader,
    calib_data_loader=val_data_loader,
    map_fit_config=FitConfig(
        monitor=FitMonitor(early_stopping_patience=2, metrics=(accuracy,))
    ),
)


# %% [markdown]
# ### Estimate predictive statistics
# We can now compute some predictive statistics by invoking the `predictive` attribute of the probabilistic classifier, and the method of interest. Most predictive statistics, e.g. mean or mode, require a loader of input data points. You can easily get this from the data loader calling its method `to_inputs_loader`.

# %% pycharm={"name": "#%%\n"}
test_log_probs = prob_model.predictive.log_prob(data_loader=test_data_loader)
test_inputs_loader = test_data_loader.to_inputs_loader()
test_means = prob_model.predictive.mean(inputs_loader=test_inputs_loader)
test_modes = prob_model.predictive.mode(
    inputs_loader=test_inputs_loader, means=test_means
)

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

# %% [markdown]
# ### Conformal prediction sets
# Fortuna allows to produce conformal prediction sets, that are sets of likely labels up to some coverage probability threshold. These can be computed starting from probability estimates obtained with or without Fortuna.

# %%
from fortuna.conformal import AdaptivePredictionConformalClassifier

val_means = prob_model.predictive.mean(inputs_loader=val_data_loader.to_inputs_loader())
conformal_sets = AdaptivePredictionConformalClassifier().conformal_set(
    val_probs=val_means,
    test_probs=test_means,
    val_targets=val_data_loader.to_array_targets(),
)

# %% [markdown]
# We can check that, on average, conformal sets for misclassified inputs are larger than for well classified ones. This confirms the intuition that the model should be more uncertain when it is wrong.

# %%
import numpy as np

avg_size = np.mean([len(s) for s in np.array(conformal_sets, dtype="object")])
avg_size_wellclassified = np.mean(
    [
        len(s)
        for s in np.array(conformal_sets, dtype="object")[test_modes == test_targets]
    ]
)
avg_size_misclassified = np.mean(
    [
        len(s)
        for s in np.array(conformal_sets, dtype="object")[test_modes != test_targets]
    ]
)
print(f"Average conformal set size: {avg_size}")
print(
    f"Average conformal set size over well classified input: {avg_size_wellclassified}"
)
print(f"Average conformal set size over misclassified input: {avg_size_misclassified}")

# %% [markdown] pycharm={"name": "#%% md\n"}
# ### What if we have model outputs to start from?

# %% [markdown] pycharm={"name": "#%% md\n"}
# If you have already trained an MNIST model and obtained model outputs, you can still use Fortuna to calibrate them, and estimate uncertainty. For educational purposes only, let us take the logarithm of the predictive mean estimated above as model outputs, and pretend these were generated with some other framework. Furthermore, we store arrays of validation and test target variables, and assume these were also given.

# %% pycharm={"name": "#%%\n"}
import numpy as np

calib_outputs = np.log(val_means)
test_outputs = np.log(test_means)

calib_targets = val_data_loader.to_array_targets()
test_targets = test_data_loader.to_array_targets()

# %% [markdown] pycharm={"name": "#%% md\n"}
# We now invoke a calibration classifier, with default temperature scaling output calibrator, and calibrate the model outputs.

# %% pycharm={"name": "#%%\n"}
from fortuna.output_calib_model import OutputCalibClassifier

calib_model = OutputCalibClassifier()
calib_status = calib_model.calibrate(
    calib_outputs=calib_outputs, calib_targets=calib_targets
)

# %% [markdown] pycharm={"name": "#%% md\n"}
# Similarly as above, we can now compute predictive statistics.

# %% pycharm={"name": "#%%\n"}
test_log_probs = calib_model.predictive.log_prob(
    outputs=test_outputs, targets=test_targets
)
test_means = calib_model.predictive.mean(outputs=test_outputs)
test_modes = calib_model.predictive.mode(outputs=test_outputs)

# %% [markdown] pycharm={"name": "#%% md\n"}
# Then one can compute metrics and conformal intervals, exactly as done above.
