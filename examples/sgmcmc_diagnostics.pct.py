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
# # Stochastic Gradient Markov chain Monte Carlo (SG-MCMC) disagnostics

# %% [markdown]
# Markov chain Monte Carlo (MCMC) methods are powerful tools for approximating the posterior distribution. Stochastic variants, such as Stochastic Gradient Hamiltonian Monte Carlo, promise rapid sampling at the cost of more biased inference. However, it has been shown that standard MCMC diagnostics fail to detect these biases. Kernel Stein discrepancy approach (KSD) with the recently proposed inverse multiquadric (IMQ) kernel [[Gorham and Mackey, 2017](https://proceedings.mlr.press/v70/gorham17a/gorham17a.pdf)] aims for  comparing biased, exact, and deterministic sample sequences, that is also particularly suitable for parallelized computation.
#
# In this notebook, we show how to assess the quality of SG-MCMC samples.

# %% [markdown]
# We create a toy example with a 2-D multivariate normal distribution. The distribution is parametrized a zero mean and a covariance matrix $\Sigma = P^{T} D P$, where $D$ is a diagonal scale matrix, and $P$ is a rotation matrix for some angle $r$.

# %%
from jax import vmap, value_and_grad
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

mu = jnp.zeros([2,])
r = np.pi / 4
D = jnp.array([2., 1.])
P = jnp.array([[jnp.cos(r), jnp.sin(r)], [-jnp.sin(r), jnp.cos(r)]])
sigma = P.T @ jnp.diag(D) @ P

# %% [markdown]
# We create a ground truth dataset, and also two more dataset with underdispersed $\mathcal{N}(0, \sqrt[5]{\Sigma})$ and overdispersed $\mathcal{N}(0, \Sigma^{3})$ samples.

# %%
N = 1_000
disp = [1/5, 1, 3]
rng = np.random.default_rng(0)
samples = np.array([rng.multivariate_normal(mu, sigma ** d, size=N) for d in disp])

# %% [markdown]
# The dataset of samples from the target distribution (in the middle) clearly aligns with confidence ellipses.

# %%
titles = ["$\sqrt[5]{\Sigma}$", "$\Sigma$", "$\Sigma^{3}$"]
_, axs = plt.subplots(1, len(samples), sharey=True, figsize=(12, 4))
for i, ax in enumerate(axs.flatten()):
    ax.axis('equal')
    ax.grid()
    ax.scatter(samples[i, :, 0], samples[i, :, 1], alpha=0.3)
    for std in range(1, 4):
        conf_ell = Ellipse(
            xy=mu, width=D[0] * std, height=D[1] * std, angle=np.rad2deg(r),
            edgecolor='black', linestyle='--', facecolor='none'
        )
        ax.add_artist(conf_ell)
    ax.set_title(titles[i])

plt.show()

# %% [markdown]
# We construct the MVN log density function for the ground truth parameters, and compute its gradients at each sample.

# %%
def log_density_fn(params, mu=mu, sigma=sigma):
    diff = params - mu
    log_density = -jnp.log(2 * jnp.pi) * mu.size / 2
    log_density -= jnp.log(jnp.linalg.det(sigma)) / 2
    log_density -= diff.T @ jnp.linalg.inv(sigma) @ diff / 2
    return log_density

_, grads = vmap(vmap(value_and_grad(log_density_fn), 0, 0), 1, 1)(samples)

# %% [markdown]
# Kernel Stein discrepancy with inverse multiquadric kernel is computed over an array of samples and corresponding gradients. Note that it has quadratic time complexity that would make it challenging to scale to large sequences.

# %%
from fortuna.prob_model.posterior.sgmcmc.sgmcmc_diagnostic import kernel_stein_discrepancy_imq

ksd = vmap(kernel_stein_discrepancy_imq, 0, 0)(samples, grads)
log_ksd = jnp.log10(ksd)

# %% [markdown]
# As expected, the lowest value of (log-) KSD is obtained in the dataset that is sampled from the ground truth distribution.

# %%
fig, ax = plt.subplots(1, 1, figsize=(6, 3))
ax.grid()
ax.plot(disp, log_ksd)
ax.set_ylabel("log KSD")
ax.set_xlabel("$\Sigma$")
plt.show()
