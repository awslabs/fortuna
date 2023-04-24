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

# # Two-moons Classification: Improved uncertainty quantification with SNGP

# %% [markdown]
# In this notebook we show how to train an [SNGP](https://arxiv.org/abs/2006.10108) model using Fortuna, showing improved
# uncertainty estimation on the two moons dataset w.r.t it's deterministic counterpart.


# %% [markdown]
# ### Download Two-Moons data from scikit-learn
# Let us first download two-moons data from [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html).

# %%

TRAIN_DATA_SIZE = 2000

from sklearn.datasets import make_moons
train_data = make_moons(n_samples=TRAIN_DATA_SIZE, noise=0.1, random_state=0)
val_data = make_moons(n_samples=500, noise=0.1, random_state=1)
test_data = make_moons(n_samples=500, noise=0.1, random_state=2)

# %% [markdown]
# ### Convert data to a compatible data loader
# Fortuna helps you converting data and data loaders into a data loader that Fortuna can digest.

# %%
from fortuna.data import DataLoader
train_data_loader = DataLoader.from_array_data(train_data, batch_size=256, shuffle=True, prefetch=True)
val_data_loader = DataLoader.from_array_data(val_data, batch_size=256, prefetch=True)
test_data_loader = DataLoader.from_array_data(test_data, batch_size=256, prefetch=True)

# %% [markdown]
# ### Define some utils for plotting the estimated uncertainty

# %%
import matplotlib.pyplot as plt
from fortuna.data import InputsLoader
import numpy as np

def plot_uncertainty(prob_model, test_data_loader, grid_size=100):
    test_inputs_loader = test_data_loader.to_inputs_loader()
    test_means = prob_model.predictive.mean(inputs_loader=test_inputs_loader)
    test_modes = prob_model.predictive.mode(inputs_loader=test_inputs_loader, means=test_means)

    fig = plt.figure(figsize=(6, 3))
    xx = np.linspace(-5, 5, grid_size)
    yy = np.linspace(-5, 5, grid_size)
    grid = np.array([[_xx, _yy] for _xx in xx for _yy in yy])
    grid_loader = InputsLoader.from_array_inputs(grid)
    grid_entropies = prob_model.predictive.entropy(grid_loader).reshape(grid_size, grid_size)
    grid = grid.reshape(grid_size, grid_size, 2)
    plt.title("Predictive uncertainty", fontsize=12)
    im = plt.pcolor(grid[:, :, 0], grid[:, :, 1], grid_entropies)
    plt.scatter(test_data[0][:, 0], test_data[0][:, 1], s=1, c=["C0" if i == 1 else "C1" for i in test_modes])
    plt.colorbar()

# %% [markdown]
# ### Define the deterministic model
# In this tutorial we will use a deep residual network.

# %%
from fortuna.model.mlp import DeepResidualNet
import flax.linen as nn

output_dim = 2
model = DeepResidualNet(
    output_dim=output_dim,
    activations=(nn.relu, nn.relu, nn.relu, nn.relu, nn.relu, nn.relu),
    widths=(128,128,128,128,128,128),
    dropout_rate=0.1,
)

# %%
from fortuna.prob_model import ProbClassifier
from fortuna.prob_model.posterior import MAPPosteriorApproximator
from fortuna.prob_model.fit_config import FitConfig, FitMonitor, FitOptimizer, FitProcessor
from fortuna.metric.classification import accuracy
import optax


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
        optimizer=FitOptimizer(method=optax.adam(1e-4), n_epochs=100),
    )
)

# %%
plot_uncertainty(prob_model, test_data_loader, grid_size=100)
plt.show()

# %% [markdown]
# ### Define the SNGP model
# Compared to the deterministic model obtained above, SNGP has two crucial differences:
#
#   1. [Spectral Normalization](https://arxiv.org/abs/1802.05957) is applied to all Dense (or Convolutional) layers.
#   2. The Dense output layers is replaced with a Gaussian Process layer.
#
# Let's see how to do it in Fortuna:

# %% [markdown]
# In order to add Spectral Normalization to a deterministic network we just need to define a new Deep Feature extractor,
# inheriting from both the feature extractor used by the deterministic model (in this case `MLPDeepFeatureExtractorSubNet`)
# and `WithSpectralNorm`. It is worth highlighting that `WithSpectralNorm` should be taken as is, while the deep feature extractor
# can be replaced with any custom object:

# %%
from fortuna.model.mlp import DeepResidualFeatureExtractorSubNet
from fortuna.model.utils.spectral_norm import WithSpectralNorm

class SNGPDeepFeatureExtractorSubNet(WithSpectralNorm, DeepResidualFeatureExtractorSubNet):
    pass

# %% [markdown]
# Then, we can define our SNGP model by:
#
# - Replacing the deep feature extractor: from `MLPDeepFeatureExtractorSubNet` to `SNGPDeepFeatureExtractorSubNet`
# - Using the `SNGPPosteriorApproximator` as the `posterior_approximator` for the `ProbModel`.
#
# Nothing else is needed, Fortuna will take care of the rest for you!
#
# %%

# %%
import jax.numpy as jnp

from fortuna.prob_model.callbacks.sngp import ResetCovarianceCallback
from fortuna.prob_model.prior import IsotropicGaussianPrior
from fortuna.prob_model.posterior.sngp.sngp_approximator import SNGPPosteriorApproximator

output_dim = 2
model = SNGPDeepFeatureExtractorSubNet(
        activations=tuple([nn.relu]*6),
        widths=tuple([128]*6),
        dropout_rate=0.1,
        spectral_norm_bound=0.9,
    )

prob_model = ProbClassifier(
    model=model,
    prior=IsotropicGaussianPrior(log_var=jnp.log(1./1e-4) - jnp.log(TRAIN_DATA_SIZE)),
    posterior_approximator=SNGPPosteriorApproximator(output_dim=output_dim),
    output_calibrator=None,
)

# %% [markdown]
# Notice that the only required argument when initializing `SNGPPosteriorApproximator` is
# `output_dim`, which should be set to the number of classes in the classification task.
# `SNGPPosteriorApproximator` has more optional parameters that you can play with, to gain a better understanding of those you can
# check out the documentation and/or the [original paper](https://arxiv.org/abs/2006.10108).
#
# %%

# %% [markdown]
# We are now ready to train the model as we usually do:
#
# %%

# %%
status = prob_model.train(
    train_data_loader=train_data_loader,
    val_data_loader=val_data_loader,
    calib_data_loader=val_data_loader,
    fit_config=FitConfig(
        monitor=FitMonitor(metrics=(accuracy,)),
        optimizer=FitOptimizer(method=optax.adam(1e-4), n_epochs=100),
    )
)

# %%
plot_uncertainty(prob_model, test_data_loader, grid_size=100)
plt.show()

# %% [markdown]

# We can clearly see that the SNGP model provides much better uncertainty estimates compared to the deterministic one.


