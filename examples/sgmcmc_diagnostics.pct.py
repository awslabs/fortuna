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
# # Stochastic Gradient Markov chain Monte Carlo (SG-MCMC) disagnostics

# %% [markdown]
# Markov chain Monte Carlo (MCMC) methods are powerful tools for approximating the posterior distribution. Stochastic procedures, such as Stochastic Gradient Hamiltonian Monte Carlo, enable rapid sampling at the cost of more biased inference. However, it has been shown that standard MCMC diagnostics fail to detect these biases. Kernel Stein discrepancy approach (KSD) with the recently proposed inverse multiquadric (IMQ) kernel [[Gorham and Mackey, 2017](https://proceedings.mlr.press/v70/gorham17a/gorham17a.pdf)] aims for comparing biased, exact, and deterministic sample sequences, that is also particularly suitable for parallelized computation.
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

mu = jnp.zeros(
    [
        2,
    ]
)
r = np.pi / 4
D = jnp.array([2.0, 1.0])
P = jnp.array([[jnp.cos(r), jnp.sin(r)], [-jnp.sin(r), jnp.cos(r)]])
sigma = P.T @ jnp.diag(D) @ P

# %% [markdown]
# We create a ground truth dataset, and also two more dataset with underdispersed $\mathcal{N}(0, \sqrt[5]{\Sigma})$ and overdispersed $\mathcal{N}(0, \Sigma^{3})$ samples.

# %%
N = 1_000
disp = [1 / 5, 1, 3]
rng = np.random.default_rng(0)
samples = np.array([rng.multivariate_normal(mu, sigma**d, size=N) for d in disp])

# %% [markdown]
# The dataset of samples from the target distribution (in the middle) clearly aligns with confidence ellipses.

# %%
titles = ["$\sqrt[5]{\Sigma}$", "$\Sigma$", "$\Sigma^{3}$"]
_, axs = plt.subplots(1, len(samples), sharey=True, figsize=(12, 4))
for i, ax in enumerate(axs.flatten()):
    ax.axis("equal")
    ax.grid()
    ax.scatter(samples[i, :, 0], samples[i, :, 1], alpha=0.3)
    for std in range(1, 4):
        conf_ell = Ellipse(
            xy=mu,
            width=D[0] * std,
            height=D[1] * std,
            angle=np.rad2deg(r),
            edgecolor="black",
            linestyle="--",
            facecolor="none",
        )
        ax.add_artist(conf_ell)
    ax.set_title(titles[i])

plt.show()

# %% [markdown]
# Kernel Stein discrepancy with inverse multiquadric kernel is computed over an array of samples and corresponding gradients. Note that it has quadratic time complexity that would make it challenging to scale to large sequences.

# %%
from fortuna.prob_model.posterior.sgmcmc.sgmcmc_diagnostic import (
    kernel_stein_discrepancy_imq,
)

logpdf = lambda params: stats.multivariate_normal.logpdf(params, mu, sigma)
_, grads = vmap(vmap(value_and_grad(logpdf), 0, 0), 1, 1)(samples)

ksd = vmap(kernel_stein_discrepancy_imq, 0, 0)(samples, grads)
log_ksd = jnp.log10(ksd)

# %% [markdown]
# As expected, the lowest value of (log-)KSD is obtained in the dataset that is sampled from the ground truth distribution.

# %%
fig, ax = plt.subplots(1, 1, figsize=(6, 3))
ax.grid()
ax.plot(disp, log_ksd)
ax.set_ylabel("log KSD")
ax.set_xlabel("$\Sigma$")
plt.show()

# %% [markdown]
# ### Estimating effective sample size
#
# Effective Sample Size (ESS) is a metric that quantifies autocorrelation in a sequence. Intuitively, ESS is the size of an i.i.d. sample with the same variance as the input sample. Typical usage includes computing the standard error for the MCMC estimator:


# %%
from fortuna.prob_model.posterior.sgmcmc.sgmcmc_diagnostic import effective_sample_size

ess = effective_sample_size(samples[0])
variance = jnp.var(samples[0], axis=0)
standard_error = jnp.sqrt(variance / ess)
standard_error

# %% [markdown]
# Note that a sequence of strongly autocorrelated samples leads to a very low ESS:

# %%
print("ESS for no auto-correlation:", effective_sample_size(rng.normal(size=200)))
print(
    "ESS for strong auto-correlation:",
    effective_sample_size(jnp.arange(200) + rng.normal(size=200)),
)
