# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Bring in your own objects

# %% [markdown]
# When constructing a probabilistic model, you can bring your own model, prior distribution and output calibrator. Let's make some examples.

# %% [markdown]
# ## Bring in your own model

# %% [markdown]
# As an example, we show how to construct an arbitrary Convolutional Neural Network (CNN) model.

# %%
import flax.linen as nn

class CNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        x = nn.log_softmax(x)
        return x


# %% [markdown]
# We now set it as a model in a probabilistic classifier.

# %%
from fortuna.prob_model.classification import ProbClassifier
prob_model = ProbClassifier(model=CNN())

# %% [markdown]
# Done. Let's check that it works by initializing its parameters and doing a forward pass.

# %%
from jax import random
import jax.numpy as jnp
x = jnp.zeros((1, 64, 64, 10))
variables = prob_model.model.init(random.PRNGKey(0), x)
prob_model.model.apply(variables, x)

# %% [markdown]
# ## Bring in your own prior distribution

# %% [markdown]
# As an example, we show how to construct a multi-dimensional uniform prior distribution.

# %%
from fortuna.prob_model.prior import Prior
from fortuna.typing import Params
from typing import Optional
from fortuna.utils.random import generate_rng_like_tree
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map
from jax._src.prng import PRNGKeyArray
import jax.numpy as jnp

class Uniform(Prior):
    def log_joint_prob(self, params: Params) -> float:
        v = jnp.mean((ravel_pytree(params)[0] <= 1) & (ravel_pytree(params)[0] >= 0))
        return jnp.where(v == 1., jnp.array(0), -jnp.inf)

    def sample(self, params_like: Params, rng: Optional[PRNGKeyArray] = None) -> Params:
        if rng is None:
            rng = self.rng.get()
        keys = generate_rng_like_tree(rng, params_like)
        return tree_map(lambda l, k: random.uniform(k, l.shape, l.dtype), params_like, keys,)


# %% [markdown]
# In the code below, we test the uniform prior we just created. In order to call `sample`, we will set `prior.rng` to a `RandomNumberGenerator` object, which automatically handles and updates random number generators starting from a random seed. This is usually automatically done by the probabilistic model, so you never need to worry about this. But in this case, since we are testing a derived class of `Prior` in isolation, we need this.

# %%
from fortuna.utils.random import RandomNumberGenerator
prior = Uniform()
prior.rng = RandomNumberGenerator(seed=0)
params_in = dict(a=jnp.array([1.]), b=jnp.array([[0.]]), c=jnp.array([0.5, 1.]))
params_out = dict(a=jnp.array([1.]), b=jnp.array([[0.]]), c=jnp.array([3., 1.]))
print(f"log-prob(params_in): {prior.log_joint_prob(params_in)}")
print(f"log-prob(params_out): {prior.log_joint_prob(params_out)}")
print(f"sample: {prior.sample(params_in)}")

# %% [markdown]
# To use your your uniform prior in Fortuna, just set it as the `prior` parameter of your `ProbClassifier` or `ProbRegressor`.

# %% [markdown]
# ## Bring in your own output calibrator

# %% [markdown]
# As an example, we show how to construct an MLP output calibrator. Mind that an output calibrator is just any Flax model, and as such you could also use the MLP pre-built in Fortuna. However, here we implement one from scratch for educational purpose.

# %%
import flax.linen as nn
from typing import Tuple

class MLP(nn.Module):
    features: Tuple[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.relu(nn.Dense(feat)(x))
            x = nn.Dense(self.features[-1])(x)
        return x


# %% [markdown]
# You can now set your MLP as the output calibrator of a probabilistic model, or a calibration model. We do it here for a calibration regressor.

# %%
from fortuna.calib_model.regression import CalibRegressor
calib_model = CalibRegressor(output_calibrator=MLP(features=(4, 2, 1)))

# %% [markdown]
# Done. Let's check that it works by initializing its parameters and doing a forward pass.

# %% pycharm={"name": "#%%\n"}
from jax import random
import jax.numpy as jnp
x = jnp.ones((1, 10))
variables = calib_model.output_calibrator.init(random.PRNGKey(0), x)
calib_model.output_calibrator.apply(variables, x)
