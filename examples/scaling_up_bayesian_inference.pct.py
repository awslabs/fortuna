# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: fortuna
#     language: python
#     name: fortuna
# ---

# # Scaling Bayesian inference up

# Bayesian inference is sometimes regarded as unfeasible on large deep learning models for mainly two reasons:
# 1. **memory**. Bayesian methods often require multiple copies of model parameters, which might not fit within GPU memory.
# 2. **Curse-of-dimensionality**. As the number of parameters increase, proper inference becomes harder and harder. 
#
# To remedy this problem, Fortuna offers the possibility to effortlessly *freeze* some model parameters, and run any Bayesian inference method only on the others. As an example, one may treat deterministically all parameters up to the second-last layer of a model, and exploit a Bayesian treatment exclusively on the last layer. 
#
# This simple strategy can be seen as a way to compromise between standard training procedures and full-blown Bayesian methods. By reducing the size of the parameters upon which Bayesian inference is performed, we simultaneously mitigate memory problems and reduce curse-of-dimensionality, suddendly bringing Bayesian inference on large models back in the game.

# Mathematically, let us denote model parameters by $\theta$ and training data by $\mathcal{D}$. If we split the model parameters $\theta$ into $\theta_{\text{frozen}}$ and $\theta_{\text{trainable}}$, a Bayesian treatment on subsets of the model parameters corresponds to approximating the posterior distribution as
#
# $$ p(\theta|\mathcal{D}) \approx \delta_{\theta_{\text{frozen}}^*}(\theta)\,\tilde{p}(\theta_{\text{trainable}}|\mathcal{D}), $$
#
# where $\theta_{\text{frozen}}^*$ is the value that the frozen parameters have been frozen to, $\delta_{\theta_{\text{frozen}}^*}(\cdot)$ denotes a Dirac delta centered at $\theta_{\text{frozen}}^*$, and $\tilde{p}$ denotes the posterior approximation given by the Bayesian method. The frozen parameters can be estimated in several ways. For this purpose, Fortuna offers out-of-the-box a Maximum-A-Posteriori (MAP) procedure, which essentially produces a regularized maximum-likelihood estimator.

# Let's see how to do all of this in a few lines of Fortuna code!

# ## Define some fake data

# Let us define a fake data loader, with the only purpose of showing how to train a Bayesian inference method while freezing part of the model.

from fortuna.data import DataLoader
from jax import random
output_dim = 10
n_data = 50
train_data_loader = DataLoader.from_array_data(
    data=(jnp.linspace(0, 10, n_data), random.choice(random.PRNGKey(0), output_dim, shape=(n_data,)))
)

# ## Define a model

# We now define a simple feedforward neural network model. Any more complex Flax model, including models already built in Fortuna, would also be suitable.

# +
from flax import linen as nn
import jax.numpy as jnp

class Model(nn.Module):
    output_dim: int

    @nn.compact
    def __call__(self, x, train: bool = False, **kwargs) -> jnp.ndarray:
        x = x.reshape(x.shape[0], -1)
        x = nn.Dense(5, name="l1")(x)
        x = nn.relu(x)
        x = nn.Dense(5, name="l2")(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim, name="l3")(x)
        return x


# -

# This model contains 3 different layers. However, we would like to do Bayesian inference only on the last one, while learning all the other with a MAP.

# ## Create a probabilistic classifier

# We now create a probabilistic classifier plugging in the model that we just created. As a Bayesian method, we will choose a Laplace approximation, but any other method would work too.

from fortuna.prob_model import ProbClassifier, LaplacePosteriorApproximator
prob_model = ProbClassifier(
    model=Model(output_dim=output_dim), 
    posterior_approximator=LaplacePosteriorApproximator()
)

# ## Train!

# We are ready to call `prob_model.train`, which will perform posterior inference under-the-hood. In order to do Bayesian inference on the last layer only and freeze the other parameters, all we need to do is to pass a function `freeze_fun` to the optimizer configuration object, deciding which parameters should be "frozen" and which should be "trainable".
#
# In addition, we configure `map_fit_config` to make a preliminary run with MAP, and set the frozen parameters to a meaningful value. Alternatively, if any of these is available, you can also either restore an existing checkpoint by configuring `FitCheckpointer.restore_checkpoint_path`, or start from a current state by setting `FitCheckpointer.start_from_current_state` to `True`. 

from fortuna.prob_model import FitConfig, FitOptimizer
status = prob_model.train(
    train_data_loader=train_data_loader,
    fit_config=FitConfig(
        optimizer=FitOptimizer(
            n_epochs=2, 
            freeze_fun=lambda path, v: "trainable" if "l3" in path else "frozen"
        )
    ),
    map_fit_config=FitConfig(optimizer=FitOptimizer(n_epochs=2))
)

# *Et voilà!*
