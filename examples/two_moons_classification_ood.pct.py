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
#   nbsphinx:
#     execute: never
# ---

# %% [markdown]
#
# # Two-moons Classification: Improved uncertainty quantification

# %% [markdown]
# In this notebook we will see how to fix model overconfidence over inputs that are far-away from the training data.
# We will do that using two different approaches; let's dive right into it!


# %% [markdown]
# ### Setup
# #### Download the Two-Moons data from scikit-learn
# Let us first download the two-moons data from [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html).

# %%
TRAIN_DATA_SIZE = 500

from sklearn.datasets import make_moons

train_data = make_moons(n_samples=TRAIN_DATA_SIZE, noise=0.1, random_state=0)
val_data = make_moons(n_samples=500, noise=0.1, random_state=1)
test_data = make_moons(n_samples=500, noise=0.1, random_state=2)

# %% [markdown]
# #### Convert data to a compatible data loader
# Fortuna helps you convert data and data loaders into a data loader that Fortuna can digest.

# %%
from fortuna.data import DataLoader

train_data_loader = DataLoader.from_array_data(
    train_data, batch_size=256, shuffle=True, prefetch=True
)
val_data_loader = DataLoader.from_array_data(val_data, batch_size=256, prefetch=True)
test_data_loader = DataLoader.from_array_data(test_data, batch_size=256, prefetch=True)

# %% [markdown]
# #### Define some utils for plotting the estimated uncertainty

# %%
import matplotlib.pyplot as plt
import numpy as np
from fortuna.data import InputsLoader
from fortuna.prob_model import ProbClassifier
import jax.numpy as jnp


def get_grid_inputs_loader(grid_size: int = 100):
    xx = np.linspace(-3, 4, grid_size)
    yy = np.linspace(-1.5, 2, grid_size)
    grid = np.array([[_xx, _yy] for _xx in xx for _yy in yy])
    grid_inputs_loader = InputsLoader.from_array_inputs(grid)
    grid = grid.reshape(grid_size, grid_size, 2)
    return grid, grid_inputs_loader


def compute_test_modes(prob_model: ProbClassifier, test_data_loader: DataLoader):
    test_inputs_loader = test_data_loader.to_inputs_loader()
    test_means = prob_model.predictive.mean(inputs_loader=test_inputs_loader)
    return prob_model.predictive.mode(
        inputs_loader=test_inputs_loader, means=test_means
    )


def plot_uncertainty_over_grid(
    grid: jnp.ndarray, scores: jnp.ndarray, test_modes: jnp.ndarray, title: str, ax=None
):
    scores = scores.reshape(grid.shape[0], grid.shape[1])
    if ax is None:
        fig, ax = plt.subplots()
    ax.set_title(title, fontsize=12)
    pcm = ax.imshow(
        scores.T,
        origin="lower",
        extent=(grid[0][0][0], grid[-1][0][0], grid[0][0][1], grid[0][-1][1]),
        interpolation="bicubic",
        aspect="auto",
    )

    # Plot training data.
    im = ax.scatter(
        test_data[0][:, 0],
        test_data[0][:, 1],
        s=3,
        c=["C0" if i == 1 else "C1" for i in test_modes],
    )
    plt.colorbar(im, ax=ax.ravel().tolist() if hasattr(ax, "ravel") else ax)


# %% [markdown]
# ### Define the deterministic model
# In this tutorial we will use a deep residual network, see `fortuna.model.mlp.DeepResidualNet` for
# more details on the model.

# %%
from fortuna.model.mlp import DeepResidualNet
import flax.linen as nn

output_dim = 2
model = DeepResidualNet(
    output_dim=output_dim,
    activations=(nn.relu, nn.relu, nn.relu, nn.relu, nn.relu, nn.relu),
    widths=(128, 128, 128, 128, 128, 128),
    dropout_rate=0.1,
)

# %%
from fortuna.prob_model import MAPPosteriorApproximator
from fortuna.prob_model import FitConfig, FitMonitor, FitOptimizer
from fortuna.metric.classification import accuracy


prob_model = ProbClassifier(
    model=model,
    posterior_approximator=MAPPosteriorApproximator(),
    output_calibrator=None,
)
status = prob_model.train(
    train_data_loader=train_data_loader,
    val_data_loader=val_data_loader,
    calib_data_loader=val_data_loader,
    fit_config=FitConfig(
        monitor=FitMonitor(metrics=(accuracy,)),
        optimizer=FitOptimizer(n_epochs=100),
    ),
)

# %%
test_modes = compute_test_modes(prob_model, test_data_loader)
grid, grid_inputs_loader = get_grid_inputs_loader(grid_size=100)
grid_entropies = prob_model.predictive.entropy(grid_inputs_loader)
plot_uncertainty_over_grid(
    grid=grid,
    scores=grid_entropies,
    test_modes=test_modes,
    title="Predictive uncertainty with MAP",
)
plt.show()

# %% [markdown]
# Clearly, the model is overconfident on inputs that are far away from the training data.
# This behaviour is not what one would expect, as we rather the model being less confident on out-of-distribution inputs.

# %% [markdown]
# ### Fit an OOD classifier to distinguish between in-distribution and out-of-distribution inputs
# Given the trained model from above, we can now use one of the models provided by Fortuna to actually improve
# the model's confidence on the out-of-distribution inputs.
# In the example below we will use two methods:
#
# - A classifier based on the **Malahanobis distance**, introduced in
# [[Lee et al., 2018]](https://proceedings.neurips.cc/paper/2018/file/abdeb6f575ac5c6676b747bca8d09cc2-Paper.pdf)
# - **Deep Deterministic Uncertainty (DDU)** [[Mukhoti et al., 2022]](https://arxiv.org/abs/2102.11582).

# %% [markdown]
# In the code block below, we define a
# `feature_extractor_subnet`, a sub-network of our previously trained model
# that transforms an input vector into an embedding vector. In this example, the feature extractor is taken as our original model
# (`DeepResidualNet`) without the output layer.

# %%
from fortuna.model.mlp import DeepResidualFeatureExtractorSubNet
import jax


feature_extractor_subnet = DeepResidualFeatureExtractorSubNet(
    dense=model.dense,
    widths=model.widths,
    activations=model.activations,
    dropout=model.dropout,
    dropout_rate=model.dropout_rate,
)


@jax.jit
def _apply(inputs, params, mutable):
    variables = {"params": params["model"]["params"]["dfe_subnet"].unfreeze()}
    if mutable is not None:
        mutable_variables = {
            k: v["dfe_subnet"].unfreeze() for k, v in mutable["model"].items()
        }
        variables.update(mutable_variables)
    return feature_extractor_subnet.apply(variables, inputs, train=False, mutable=False)


# %% [markdown]
# Let's use the feature extractor to get the embeddings of the training and OOD inputs.

# %%
from typing import Tuple

import tqdm

from fortuna.data.loader.base import BaseDataLoaderABC, BaseInputsLoader
from fortuna.prob_model.posterior.state import PosteriorState
from fortuna.typing import Array


def get_embeddings_and_targets(
    state: PosteriorState, train_data_loader: BaseDataLoaderABC
) -> Tuple[Array, Array]:
    train_labels = []
    train_embeddings = []
    for x, y in tqdm.tqdm(train_data_loader, desc="Computing embeddings: "):
        train_embeddings.append(
            _apply(inputs=x, params=state.params, mutable=state.mutable)
        )
        train_labels.append(y)
    train_embeddings = jnp.concatenate(train_embeddings, 0)
    train_labels = jnp.concatenate(train_labels)
    return train_embeddings, train_labels


def get_embeddings(state: PosteriorState, inputs_loader: BaseInputsLoader):
    return jnp.concatenate(
        [
            _apply(inputs=x, params=state.params, mutable=state.mutable)
            for x in inputs_loader
        ],
        0,
    )


state = prob_model.posterior.state.get()
train_embeddings, train_labels = get_embeddings_and_targets(
    state=state, train_data_loader=train_data_loader
)

# %%
from fortuna.ood_detection import (
    MalahanobisOODClassifier,
    DeepDeterministicUncertaintyOODClassifier,
)

maha_classifier = MalahanobisOODClassifier(num_classes=2)
maha_classifier.fit(embeddings=train_embeddings, targets=train_labels)

ddu_classifier = DeepDeterministicUncertaintyOODClassifier(num_classes=2)
ddu_classifier.fit(embeddings=train_embeddings, targets=train_labels)

# %% [markdown]
# Let's plot the results! For the sake of visualization, we set a threshold on the OOD classifiers scores using the maximum score obtained from a known in-distribution source.

# %%
grid, grid_inputs_loader = get_grid_inputs_loader(grid_size=100)
grid_embeddings = get_embeddings(state=state, inputs_loader=grid_inputs_loader)

ind_embeddings = get_embeddings(
    state=state, inputs_loader=val_data_loader.to_inputs_loader()
)

ind_maha_scores = maha_classifier.score(embeddings=ind_embeddings)
grid_maha_scores = maha_classifier.score(embeddings=grid_embeddings)
maha_threshold = 2 * ind_maha_scores.max()
grid_maha_scores = jnp.where(
    grid_maha_scores < maha_threshold, grid_maha_scores, maha_threshold
)

ind_ddu_scores = maha_classifier.score(embeddings=ind_embeddings)
grid_ddu_scores = ddu_classifier.score(embeddings=grid_embeddings)
ddu_threshold = 2 * ind_ddu_scores.max()
grid_ddu_scores = jnp.where(
    grid_ddu_scores < ddu_threshold, grid_ddu_scores, ddu_threshold
)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
plot_uncertainty_over_grid(
    grid=grid,
    scores=grid_maha_scores,
    test_modes=test_modes,
    title="Mahalanobis OOD scores",
    ax=axes[0],
)
plot_uncertainty_over_grid(
    grid=grid,
    scores=grid_ddu_scores,
    test_modes=test_modes,
    title="DDU OOD scores",
    ax=axes[1],
)
plt.tight_layout()
plt.show()

# %% [markdown]
# Both methods improve oveconfidence out-of-distribution! While the Mahalanobis distance is not able to remedy overconfidence closeby and between the moons, DDU manages to remedy this too.


# %% [markdown]
# ### The SNGP model
# We will now explore a different method designed to remedy overconfidence OOD, namely **Spectral Normalization Gaussian Process (SNGP)** [[Liu et al., 2020]](https://arxiv.org/abs/2006.10108).
#
# SNGP is characterized by two main features:
#
#   1. A spectral normalization is applied to all Dense (or Convolutional) layers of the deep learning model.
#   2. The Dense output layer is replaced with a Gaussian Process layer.
#
# Let's see how use SNGP in Fortuna.

# %% [markdown]
# In order to add Spectral Normalization to a deterministic network we just need to define a new deep feature extractor,
# inheriting from both the feature extractor used by the deterministic model (in this case `MLPDeepFeatureExtractorSubNet`)
# and `WithSpectralNorm`. It is worth highlighting that `WithSpectralNorm` should be taken as is, while the deep feature extractor
# can be replaced with any custom object:

# %%
from fortuna.model.mlp import DeepResidualFeatureExtractorSubNet
from fortuna.model.utils.spectral_norm import WithSpectralNorm


class SNGPDeepFeatureExtractorSubNet(
    WithSpectralNorm, DeepResidualFeatureExtractorSubNet
):
    pass


# %% [markdown]
# Then, we can define our SNGP model by:
#
# - Replacing the deep feature extractor: from `MLPDeepFeatureExtractorSubNet` to `SNGPDeepFeatureExtractorSubNet`
# - Using the `SNGPPosteriorApproximator` as the `posterior_approximator` for the `ProbModel`.
#
# Nothing else is needed, Fortuna will take care of the rest for you!

# %%
import jax.numpy as jnp

from fortuna.prob_model.prior import IsotropicGaussianPrior
from fortuna.prob_model import SNGPPosteriorApproximator

output_dim = 2
model = SNGPDeepFeatureExtractorSubNet(
    activations=tuple([nn.relu] * 6),
    widths=tuple([128] * 6),
    dropout_rate=0.1,
    spectral_norm_bound=0.9,
)

prob_model = ProbClassifier(
    model=model,
    prior=IsotropicGaussianPrior(
        log_var=jnp.log(1.0 / 1e-4) - jnp.log(TRAIN_DATA_SIZE)
    ),
    posterior_approximator=SNGPPosteriorApproximator(output_dim=output_dim),
    output_calibrator=None,
)

# %% [markdown]
# The only required argument when initializing `SNGPPosteriorApproximator` is
# `output_dim`, which should be set to the number of classes in the classification task. Other hyperparameters can be set to further improve performance - for a better understanding of these, check [[Liu et al., 2020]](https://arxiv.org/abs/2006.10108).

# %% [markdown]
# We are now ready to train the model as usual:

# %%
status = prob_model.train(
    train_data_loader=train_data_loader,
    val_data_loader=val_data_loader,
    calib_data_loader=val_data_loader,
    fit_config=FitConfig(
        monitor=FitMonitor(metrics=(accuracy,)),
        optimizer=FitOptimizer(n_epochs=100),
    ),
)

# %%
test_sngp_modes = compute_test_modes(prob_model, test_data_loader)
grid, grid_inputs_loader = get_grid_inputs_loader(grid_size=100)
grid_sngp_entropies = prob_model.predictive.entropy(grid_inputs_loader)
plot_uncertainty_over_grid(
    grid=grid,
    scores=grid_sngp_entropies,
    test_modes=test_sngp_modes,
    title="Predictive uncertainty with SNGP",
)
plt.show()

# %% [markdown]
# Similarly to the Mahalanobis and DDU methods above, SNGP also manages to remedy overconfidence. The uncertainty provided by SNGP appears similarly good to the scores obtain from DDU above, but smoother and less prone to overfitting.

# %% [markdown]
# The following figure compares all the figures obtained above in one place.

# %%
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
plot_uncertainty_over_grid(
    grid=grid,
    scores=grid_entropies,
    test_modes=test_modes,
    title="Predictive uncertainty with MAP",
    ax=axes[0],
)
plot_uncertainty_over_grid(
    grid=grid,
    scores=grid_maha_scores,
    test_modes=test_modes,
    title="Mahalanobis OOD scores",
    ax=axes[1],
)
plot_uncertainty_over_grid(
    grid=grid,
    scores=grid_ddu_scores,
    test_modes=test_modes,
    title="DDU OOD scores",
    ax=axes[2],
)
plot_uncertainty_over_grid(
    grid=grid,
    scores=grid_sngp_entropies,
    test_modes=test_sngp_modes,
    title="Predictive uncertainty with SNGP",
    ax=axes[3],
)
plt.tight_layout()
plt.show()

# %%
