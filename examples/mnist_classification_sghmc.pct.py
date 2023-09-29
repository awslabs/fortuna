# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # MNIST Classification with Stochastic Gradient Hamiltonian Monte Carlo (SGHMC)

# %% [markdown]
# In this notebook we demonstrate how to use Fortuna to obtain predictions uncertainty estimates from a simple neural network model trained for MNIST classification task, using the [SGHMC](http://proceedings.mlr.press/v32/cheni14.pdf) method.

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
# Let us build a probabilistic classifier. This is an interface object containing several attributes that you can configure, i.e. `model`, `prior`, and `posterior_approximator`. In this example, we use a multilayer perceptron and an SGHMC posterior approximator. SGHMC (and SGMCMC methods, broadly) allows configuring a step size schedule function. For simplicity, we create a constant step schedule.

# %%
import flax.linen as nn

from fortuna.prob_model import ProbClassifier, SGHMCPosteriorApproximator
from fortuna.model import MLP

output_dim = 10
prob_model = ProbClassifier(
    model=MLP(output_dim=output_dim, activations=(nn.tanh, nn.tanh)),
    posterior_approximator=SGHMCPosteriorApproximator(
        burnin_length=300, step_schedule=4e-6
    ),
)


# %% [markdown]
# ### Train the probabilistic model: posterior fitting and calibration
# We can now train the probabilistic model. This includes fitting the posterior distribution and calibrating the probabilistic model. We set the Markov chain burn-in phase to 20 epochs, followed by obtaining samples from the approximated posterior.

# %%
from fortuna.prob_model import FitConfig, FitMonitor, FitOptimizer
from fortuna.metric.classification import accuracy

status = prob_model.train(
    train_data_loader=train_data_loader,
    val_data_loader=val_data_loader,
    calib_data_loader=val_data_loader,
    fit_config=FitConfig(
        monitor=FitMonitor(metrics=(accuracy,)),
        optimizer=FitOptimizer(n_epochs=30),
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
    error=0.05,
)

# %% [markdown]
# We can check that, on average, conformal sets for misclassified inputs are larger than for well classified ones.

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

# %% [markdown]
# Furthermore, we visualize some of the examples with the largest and the smallest conformal sets. Intutively, they correspond to the inputs where the model is the most uncertain or the most certain about its predictions.

# %%
from matplotlib import pyplot as plt

N_EXAMPLES = 10
images = test_data_loader.to_array_inputs()


def visualize_examples(indices, n_examples=N_EXAMPLES):
    n_rows = min(len(indices), n_examples)
    _, axs = plt.subplots(1, n_rows, figsize=(10, 2))
    axs = axs.flatten()
    for i, ax in enumerate(axs):
        ax.imshow(images[indices[i]], cmap="gray")
        ax.axis("off")
    plt.show()


# %%
indices = np.argsort([len(s) for s in np.array(conformal_sets, dtype="object")])

# %%
print("Examples with the smallest conformal sets:")
visualize_examples(indices[:N_EXAMPLES])

# %%
print("Examples with the largest conformal sets:")
visualize_examples(np.flip(indices[-N_EXAMPLES:]))
