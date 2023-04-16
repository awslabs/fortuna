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
train_data_loader = DataLoader.from_array_data(train_data, batch_size=128, shuffle=True, prefetch=True)
val_data_loader = DataLoader.from_array_data(val_data, batch_size=128, prefetch=True)
test_data_loader = DataLoader.from_array_data(test_data, batch_size=128, prefetch=True)

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
from fortuna.model.mlp import MLP
import flax.linen as nn

output_dim = 2
model = MLP(
    output_dim=output_dim,
    activations=(nn.relu, nn.relu, nn.relu, nn.relu, nn.relu, nn.relu),
    widths=(128,128,128,128,128,128),
    dropout_rate=0.1,
)

# %%
from fortuna.prob_model import ProbClassifier
from fortuna.prob_model.posterior import MAPPosteriorApproximator
from fortuna.prob_model.fit_config import FitConfig, FitMonitor, FitOptimizer
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
# Compared to the determnistic model obtained above, SNGP has two crucial differences:
#
#   1. [Spectral Normalization](https://arxiv.org/abs/1802.05957) is applied to all Dense (or Convolutional) layers.
#   2. The Dense output layers is replaced with a Gaussian Process layer.
#
# Let's see how to do it in Fortuna:

# %% [markdown]
# In Fortuna, models are made of two main componets:
#
# - A deep feature extractor
# - An output layer
#
# In order to add Spectral Normalization to a deterministic network we just need to define a new Deep Feature extractor,
# inheriting from both the feature extractor used by the deterministic model (in this case `MLPDeepFeatureExtractorSubNet`)
# and `WithSpectralNorm`:

# %%
from fortuna.model.mlp import MLPDeepFeatureExtractorSubNet
from fortuna.model.utils.spectral_norm import WithSpectralNorm

class SNGPDeepFeatureExtractorSubNet(WithSpectralNorm, MLPDeepFeatureExtractorSubNet):
    pass

# %% [markdown]
# Then, we can define our SNGP model by:
#
# - Changing the deep feature extractor: from `MLPDeepFeatureExtractorSubNet` to `SNGPDeepFeatureExtractorSubNet`
# - Changing the output layer: from `Dense` to `RandomFeatureGaussianProcess`.

# %%
from fortuna.model.utils.random_features import RandomFeatureGaussianProcess

from fortuna.model.sngp import SNGPMixin
class SNGPModel(SNGPMixin, MLP):
    def setup(self):
        if len(self.widths) != len(self.activations):
            raise Exception(
                "`widths` and `activations` must have the same number of elements."
            )
        self.dfe_subnet = SNGPDeepFeatureExtractorSubNet(
            dense=self.dense,
            widths=self.widths,
            activations=self.activations[:-1],
            dropout=self.dropout,
            dropout_rate=self.dropout_rate,
            spectral_norm_bound=self.spectral_norm_bound,
            spectral_norm_iteration=self.spectral_norm_iteration,
        )
        self.output_subnet = RandomFeatureGaussianProcess(
            features=self.output_dim,
            hidden_features=self.gp_hidden_features,
            normalize_input=self.normalize_input,
            covmat_kwargs={
            "ridge_penalty": self.ridge_penalty,
            "momentum": self.momentum,
        },
        )

    def __call__(self, x, train=False, **kwargs):
        x = self.dfe_subnet(x, train)
        x = self.output_subnet(x, return_full_covmat=self.use_full_covmat)
        return x

# %%
model = SNGPModel(
        output_dim=output_dim,
        activations=(nn.relu, nn.relu, nn.relu, nn.relu, nn.relu, nn.relu),
        widths=(128,128,128,128,128,128),
        dropout_rate=0.1,
        gp_hidden_features=1024,
        spectral_norm_bound=0.9,
        ridge_penalty=1,
    )

# %%
from functools import partial
import jax.numpy as jnp

from fortuna.model.model_manager.classification import SNGPModelManager
from fortuna.prob_model.callbacks.sngp import ResetCovarianceCallback
from fortuna.prob_model.prior import IsotropicGaussianPrior

prob_model = ProbClassifier(
    model=model,
    prior=IsotropicGaussianPrior(log_var=jnp.log(1./1e-4) - jnp.log(TRAIN_DATA_SIZE)),
    posterior_approximator=MAPPosteriorApproximator(),
    output_calibrator=None,
    model_manager_cls=partial(SNGPModelManager, mean_field_factor=1),
)
status = prob_model.train(
    train_data_loader=train_data_loader,
    val_data_loader=val_data_loader,
    calib_data_loader=val_data_loader,
    fit_config=FitConfig(
        monitor=FitMonitor(metrics=(accuracy,)),
        optimizer=FitOptimizer(method=optax.adam(1e-4), n_epochs=100),
        callbacks=[ResetCovarianceCallback(
            precision_matrix_key_name='precision_matrix',
            ridge_penalty=1
        )]
    )
)

# %%
plot_uncertainty(prob_model, test_data_loader, grid_size=100)
plt.show()

# %% [markdown]

# We can clearly see that the SNGP model provides much better uncertainty estimates compared to the deterministic one.


