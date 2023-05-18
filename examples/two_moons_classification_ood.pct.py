# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: python3
#     language: python
#     name: python3
# ---

# %% [markdown]

# # Two-moons Classification: Improved uncertainty quantification

# %% [markdown]
# In this notebook we will see how to fix model overconfidence over inputs that are far-away from the training data.
# We will do that using two different approaches; let's dive right into it!


# %% [markdown]
# ### Setup
# #### Download the Two-Moons data from scikit-learn
# Let us first download the two-moons data from [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html).

# %%
from matplotlib import colors

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
    xx = np.linspace(-4, 4, grid_size)
    yy = np.linspace(-4, 4, grid_size)
    grid = np.array([[_xx, _yy] for _xx in xx for _yy in yy])
    grid_inputs_loader = InputsLoader.from_array_inputs(grid)
    grid = grid.reshape(grid_size, grid_size, 2)
    return grid, grid_inputs_loader


def compute_test_modes(
    prob_model: ProbClassifier, test_data_loader: DataLoader
):
    test_inputs_loader = test_data_loader.to_inputs_loader()
    test_means = prob_model.predictive.mean(inputs_loader=test_inputs_loader)
    return  prob_model.predictive.mode(
        inputs_loader=test_inputs_loader, means=test_means
    )

def plot_uncertainty_over_grid(
        grid: jnp.ndarray, scores: jnp.ndarray, test_modes: jnp.ndarray, title: str = "Predictive uncertainty"
):
    scores = scores.reshape(grid.shape[0], grid.shape[1])

    _, ax = plt.subplots(figsize=(7, 5.5))
    plt.title(title, fontsize=12)
    pcm = ax.imshow(
        scores.T,
        origin="lower",
        extent=(-4., 4., -4., 4.),
        interpolation='bicubic',
        aspect='auto')

    # Plot training data.
    plt.scatter(
        test_data[0][:, 0],
        test_data[0][:, 1],
        s=1,
        c=["C0" if i == 1 else "C1" for i in test_modes],
    )
    plt.colorbar()


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
plot_uncertainty_over_grid(grid=grid, scores=grid_entropies, test_modes=test_modes)
plt.show()

# %% [markdown]
# Clearly, the model is overconfident on inputs that are far away from the training data.
# This behaviour is not what one would expect, as we rather the model being less confident on out-of-distributin inputs.

# %% [markdown]
# ### Fit an OOD classifier to distinguish between in-distribution and out-of-distribution inputs
# Given the trained model from above, we can now use one of the models provided by Fortuna to actually improve
# the model's confidence on the out-of-distribution inputs.
# In the example below we will use the Malahanobis-based classifier introduced in
# [Lee, Kimin, et al](https://proceedings.neurips.cc/paper/2018/file/abdeb6f575ac5c6676b747bca8d09cc2-Paper.pdf)

# %%
from fortuna.ood_detection.mahalanobis import MalahanobisClassifierABC
from fortuna.model.mlp import DeepResidualFeatureExtractorSubNet
from functools import partial
import jax


class MalahanobisClassifier(MalahanobisClassifierABC):
    @partial(jax.jit, static_argnums=(0,))
    def apply(self, inputs, params, mutable, **kwargs):
        variables = {'params': params["model"]['params']['dfe_subnet'].unfreeze()}
        if mutable is not None:
            mutable_variables = {k: v['dfe_subnet'].unfreeze() for k, v in mutable["model"].items()}
            variables.update(mutable_variables)
        return self.feature_extractor_subnet.apply(
                variables, inputs, train=False, mutable=False,
            )

ood_classifier = MalahanobisClassifier(
    feature_extractor_subnet=DeepResidualFeatureExtractorSubNet(
            dense=model.dense,
            widths=model.widths,
            activations=model.activations,
            dropout=model.dropout,
            dropout_rate=model.dropout_rate,
        )
)

# %% [markdown]
# In the code block above we first define a `MalahanobisClassifier` starting from the `MalahanobisClassifierABC`
# provided by Fortuna. The only thing we need to do here is to implement the `apply` method that allow one to transform
# an input vector into an embedding vector.
# Once this is done, we can initialize our classifier by providing the `feature_extractor_subnet`. In the example,
# this is our original model (`DeepResidualNet`) without the output layer.
# We are now ready to fit the classifier using our training data and verify whether the model's overconfidence has been
# (at least partially) fixed:

# %%
state = prob_model.posterior.state.get()
ood_classifier.fit(state=state, train_data_loader=train_data_loader, num_classes=2)
grid, grid_inputs_loader = get_grid_inputs_loader(grid_size=100)
grid_scores = ood_classifier.score(state=state, inputs_loader=grid_inputs_loader)
# for the sake of plotting we set a threshold on the OOD classifier scores using the max score
# obtained from a known in-distribution source
ind_scores = ood_classifier.score(state=state, inputs_loader=val_data_loader.to_inputs_loader())
threshold = ind_scores.max()*2
grid_scores = jnp.where(grid_scores < threshold, grid_scores, threshold)
plot_uncertainty_over_grid(grid=grid, scores=grid_scores, test_modes=test_modes, title="OOD scores")
plt.show()


# %% [markdown]
# We will now see a different way of obtaining improved uncertainty estimation
# (for out-of-distribution inputs): [SNGP](https://arxiv.org/abs/2006.10108).
# Unlike before, we now have to retrain the model as the architecture will slighly change.
# The reason for this will be clear from the model definition below.

# %% [markdown]
# ### Define the SNGP model
# Compared to the deterministic model obtained in the first part of this notebook, SNGP has two crucial differences:
#
#   1. [Spectral Normalization](https://arxiv.org/abs/1802.05957) is applied to all Dense (or Convolutional) layers.
#   2. The Dense output layer is replaced with a Gaussian Process layer.
#
# Let's see how to do it in Fortuna:

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
# Notice that the only required argument when initializing `SNGPPosteriorApproximator` is
# `output_dim`, which should be set to the number of classes in the classification task.
# `SNGPPosteriorApproximator` has more optional parameters that you can play with, to gain a better understanding of those you can
# check out the documentation and/or the [original paper](https://arxiv.org/abs/2006.10108).

# %% [markdown]
# We are now ready to train the model as we usually do:

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
test_modes = compute_test_modes(prob_model, test_data_loader)
grid, grid_inputs_loader = get_grid_inputs_loader(grid_size=100)
grid_entropies = prob_model.predictive.entropy(grid_inputs_loader)
plot_uncertainty_over_grid(grid=grid, scores=grid_entropies, test_modes=test_modes)
plt.show()

# %% [markdown]

# We can clearly see that the SNGP model provides much better uncertainty estimates compared to the deterministic one.
