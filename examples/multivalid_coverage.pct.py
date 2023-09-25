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
# # Batch Multivalid Conformal Prediction (BatchMVP)

# %% [markdown]
# In this notebook we showcase the usage of BatchMVP [[Jung C. et al., 2022]](https://arxiv.org/pdf/2209.15145.pdf), a conformal prediction algorithm that satisfies coverage guarantees conditioned on group membership and non-conformity thresholds.
#
# To make it concrete, suppose that, for your application, you do not only care about marginal calibration over the full input domain, say $\mathbb{R}$, but you specifically want to ensure marginal coverage on both sub-domains $\mathbb{R}_-$ and $\mathbb{R}_+$. This property is usually not guaranteed by standard conformal prediction methods that satisfy only marginal coverage. In fact, a method may overcover on $\mathbb{R}_-$ and undercover on $\mathbb{R}_+$, and yet satisfy marginal coverage overall.
#
# We study this problem exactly in a simple regression setting, where data in $\mathbb{R}_+$ is far more noisy than in $\mathbb{R}_-$. The next cell provides functionality to generate the data and plot some intervals.

# %%
import numpy as np


def generate_data(n_data: int, sigma1=0.03, sigma2=0.5, seed: int = 43):
    rng = np.random.default_rng(seed=seed)
    x = np.concatenate(
        [
            rng.normal(loc=-1, scale=0.3, size=(n_data // 2, 1)),
            rng.normal(loc=1, scale=0.3, size=(n_data - n_data // 2, 1)),
        ]
    )
    y = np.cos(x) + np.concatenate(
        [
            rng.normal(scale=sigma1, size=(n_data // 2, 1)),
            rng.normal(scale=sigma2, size=(n_data - n_data // 2, 1)),
        ]
    )
    return x, y


def plot_intervals(xx, means, intervals, test_data, method):
    plt.figure(figsize=(6, 3))
    plt.plot(xx, xx_means, label="predictions")
    plt.scatter(*test_data, label="test data", c="C3", s=1)
    plt.fill_between(
        xx.squeeze(1),
        intervals[:, 0],
        intervals[:, 1],
        alpha=0.3,
        color="C0",
        label="95% intervals",
    )
    plt.vlines(
        0, -1, 1.5, linestyle="--", color="black", label="groups: x < 0 and x > 0"
    )
    plt.ylim([-1, 1.5])
    plt.title(method)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="lower left")
    plt.grid()


# %% [markdown]
# ## Generate and prepare the data

# %% [markdown]
# Let us first generate training, calibration and test data points.

# %%
train_data = generate_data(500)
calib_data = generate_data(500)
test_data = generate_data(500)

# %% [markdown]
# We then plot the training data. We see that when $x<0$ the data is much less noisy than when $x>0$.

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 3))
plt.scatter(*train_data, s=1)
plt.xlabel("x")
plt.ylabel("y")
plt.grid()

# %% [markdown]
# We then use Fortuna to transform these arrays of data into a data loader, in order to be able to feed data to algorithms in batches.

# %%
from fortuna.data import DataLoader

train_data_loader = DataLoader.from_array_data(
    train_data, batch_size=128, shuffle=True, prefetch=True
)
calib_data_loader = DataLoader.from_array_data(
    calib_data, batch_size=128, prefetch=True
)
test_data_loader = DataLoader.from_array_data(test_data, batch_size=128, prefetch=True)

test_inputs_loader = test_data_loader.to_inputs_loader()
test_targets = test_data_loader.to_array_targets()
calib_inputs_loader = calib_data_loader.to_inputs_loader()
calib_targets = calib_data_loader.to_array_targets()

# %% [markdown]
# ## Groups definition

# %% [markdown]
# We now rigorously define the two groups of inputs that we are going to consider, specifically $\mathbb{R}_-$ and $\mathbb{R}_+$. We define these in terms of boolean group functions. These will be used later by `BatchMVP`.

# %%
group_fns = [lambda x: x.squeeze(1) < 0, lambda x: x.squeeze(1) >= 0]
idx_left, idx_right = [group_fns[i](test_data[0]) for i in range(2)]

# %% [markdown]
# ## Credible intervals with SWAG

# %% [markdown]
# We start by training a probabilistic classifier based on a MultiLayer Perceptron (MLP) model. Under the hood, posterior inference is performed by SWAG [Maddox W. J. et al., 2019](https://proceedings.neurips.cc/paper/2019/hash/118921efba23fc329e6560b27861f0c2-Abstract.html).

# %%
from fortuna.prob_model import ProbRegressor, FitConfig, FitMonitor
from fortuna.model import MLP

prob_model = ProbRegressor(model=MLP(1), likelihood_log_variance_model=MLP(1))
status = prob_model.train(
    train_data_loader,
    map_fit_config=FitConfig(monitor=FitMonitor(early_stopping_patience=2)),
)

# %% [markdown]
# We then compute predictive mean and 95% credible intervals, and we measure marginal coverage on the full domain and on each of the two groups. We notice that the method tend to undercover overall and also on each group, compared to the desired coverage of 95%. Better hyper-parameter configurations and generating more samples to compute statistics may help achieve a better coverage already, but for the purpose of this example we will directly look at how to calibrate it with conformal prediction methods.

# %%
from fortuna.metric.regression import picp

test_means = prob_model.predictive.mean(test_inputs_loader)
test_cred_intervals = prob_model.predictive.credible_interval(test_inputs_loader)

cred_coverage = picp(*test_cred_intervals.T, test_targets)
cred_coverage_left = picp(*test_cred_intervals[idx_left].T, test_targets[idx_left])
cred_coverage_right = picp(*test_cred_intervals[idx_right].T, test_targets[idx_right])
print(f"Estimated marginal coverage of SWAG: {cred_coverage}.")
print(f"Estimated coverage of SWAG for negative inputs: {cred_coverage_left}.")
print(f"Estimated coverage of SWAG for positive inputs: {cred_coverage_right}.")

# %% [markdown]
# But first, let us visualize predictions and uncertainty! Although the metrics above told us that we are slightly undercovering, uncertainty estimates look quite good already.

# %%
from fortuna.data import InputsLoader

xx = np.linspace(test_data[0].min(), test_data[0].max())[:, None]
xx_loader = InputsLoader.from_array_inputs(xx)
xx_means = prob_model.predictive.mean(xx_loader)

# %%
xx_cred_intervals = prob_model.predictive.credible_interval(xx_loader)
plot_intervals(xx, xx_means, xx_cred_intervals, test_data, "SWAG")

# %% [markdown]
# ## Conformalized Quantile Regression

# %% [markdown]
# Starting from the credible intervals given by SWAG, we now apply Conformalized Quantile Regression (CQR) [[Romano Y. et al., 2019]](https://proceedings.neurips.cc/paper/2019/file/5103c3584b063c431bd1268e9b5e76fb-Paper.pdf) as a *post-hoc* calibration method to improve the coverage.

# %%
from fortuna.conformal import QuantileConformalRegressor

calib_cred_intervals = prob_model.predictive.credible_interval(
    inputs_loader=calib_inputs_loader
)
test_qcr_intervals = QuantileConformalRegressor().conformal_interval(
    *calib_cred_intervals.T, *test_cred_intervals.T, calib_targets, error=0.05
)

# %% [markdown]
# The following cell shows that yes, the overall marginal coverage is now closer to the desired 95%. However, CQR tends to overcover for negative inputs and undercover for positive inputs. This is not surprising, as CQR does not necessarily satisfy group-conditional coverage.

# %%
qcr_coverage = picp(*test_qcr_intervals.T, test_targets)
idx_left, idx_right = [group_fns[i](test_data[0]) for i in range(2)]
qcr_coverage_left = picp(*test_qcr_intervals[idx_left].T, test_targets[idx_left])
qcr_coverage_right = picp(*test_qcr_intervals[idx_right].T, test_targets[idx_right])
print(f"Estimated marginal coverage with CQR: {qcr_coverage}.")
print(f"Estimated coverage for the negative inputs with CQR: {qcr_coverage_left}.")
print(f"Estimated coverage for the positive inputs with CQR: {qcr_coverage_right}.")

# %% [markdown]
# Again, we visualize the uncertainty. Notice that, for the negative inputs, the coverage is over the desired 95%, while for positive inputs it is a little too low.

# %%
xx_qcr_intervals = QuantileConformalRegressor().conformal_interval(
    *calib_cred_intervals.T, *xx_cred_intervals.T, calib_targets, error=0.05
)
plot_intervals(xx, xx_means, xx_qcr_intervals, test_data, "CQR")

# %% [markdown]
# # Batch MVP

# %% [markdown]
# We finally introduce Batch MVP [[Jung C. et al., 2022]](https://arxiv.org/pdf/2209.15145.pdf) and show that it improves group-conditional coverage. For its usage, we require:
#
# - non-conformity scores evaluated on calibration. These can be evaluations of any score function measuring the degree of non-conformity between inputs $x$ and targets $y$. The less $x$ and $y$ conform with each other, the larger the score should be. A simple example of score function in regression is $s(x,y)=|y - h(x)|$, where $h$ is an arbitrary model. For the purpose of this example, we use the same score function as in CQR, that is $s(x,y)=\max\left(q_{\frac{\alpha}{2}} - y, y - q_{1 - \frac{\alpha}{2}}\right)$, where $\alpha$ is the desired coverage error, i.e. $\alpha=0.05$, and $q_\alpha$ is a corresponding quantile.
#
# - group evaluations on calibration and test data. These construct sub-domains of interest of the input domain. As we defined above, here we use $g_1(x) = \mathbb{1}[x < 0]$ and $g_2(x) = \mathbb{1}[x \ge 0]$.
#
# That's it! Defined these, we are ready to run `BatchMVP`.

# %%
from fortuna.conformal.regression.batch_mvp import BatchMVPConformalRegressor
import jax.numpy as jnp

qleft, qright = prob_model.predictive.quantile(
    [0.05 / 2, 1 - 0.05 / 2], calib_inputs_loader
)
scores = jnp.maximum(qleft - calib_targets, calib_targets - qright).squeeze(1)
min_score, max_score = scores.min(), scores.max()
scores = (scores - min_score) / (max_score - min_score)
groups = jnp.stack([g(calib_data[0]) for g in group_fns], axis=1)
test_groups = jnp.stack([g(test_data[0]) for g in group_fns], axis=1)

batchmvp = BatchMVPConformalRegressor()
test_thresholds, status = batchmvp.calibrate(
    scores=scores,
    groups=groups,
    test_groups=test_groups,
)
test_thresholds = min_score + (max_score - min_score) * test_thresholds

# %% [markdown]
# At each iteration, `BatchMVP` we compute the maximum calibration error over the different groups. We report its decay in the following picture.

# %%
plt.figure(figsize=(6, 3))
plt.plot(status["losses"], label="mean squared error decay")
plt.xlabel("rounds")
plt.legend()
plt.show()

# %% [markdown]
# Given the test thresholds, we can find the lower and upper bounds of the conformal intervals by inverting the score function $s(x, y)$ with respect to $y$. This gives $b(x, \tau) = [q_{\frac{\alpha}{2}}(x) - \tau, q_{1 - \frac{\alpha}{2}}(x) + \tau]$, where $\tau$ denotes the thresholds.

# %%
test_qleft, test_qright = prob_model.predictive.quantile(
    [0.05 / 2, 1 - 0.05 / 2], test_inputs_loader
)
test_qleft, test_qright = test_qleft.squeeze(1), test_qright.squeeze(1)
test_batchmvp_intervals = jnp.stack(
    (test_qleft - test_thresholds, test_qright + test_thresholds), axis=1
)

# %% [markdown]
# We now compute coverage metrics. As expected, `BatchMVP` not only provides a good marginal coverage overall, but also improves coverage on both negative and positive inputs.

# %%
batchmvp_coverage = picp(*test_batchmvp_intervals.T, test_targets)
batchmvp_coverage_left = picp(
    *test_batchmvp_intervals[idx_left].T, test_targets[idx_left]
)
batchmvp_coverage_right = picp(
    *test_batchmvp_intervals[idx_right].T, test_targets[idx_right]
)
print(f"Estimated marginal coverage of BatchMVP: {batchmvp_coverage}.")
print(f"Estimated coverage of BatchMVP for negative inputs: {batchmvp_coverage_left}.")
print(f"Estimated coverage of BatchMVP for positive inputs: {batchmvp_coverage_right}.")

# %% [markdown]
# Once again, we visualize predictions and estimated intervals.

# %%
xx_qleft, xx_qright = prob_model.predictive.quantile(
    [0.05 / 2, 1 - 0.05 / 2], InputsLoader.from_array_inputs(xx)
)
xx_qleft, xx_qright = xx_qleft.squeeze(1), xx_qright.squeeze(1)
xx_groups = jnp.stack([g(xx) for g in group_fns], axis=1)
xx_thresholds = batchmvp.apply_patches(groups=xx_groups)
xx_thresholds = min_score + (max_score - min_score) * xx_thresholds
xx_batchmvp_intervals = jnp.stack(
    (xx_qleft - xx_thresholds, xx_qright + xx_thresholds), axis=1
)
plot_intervals(xx, xx_means, xx_batchmvp_intervals, test_data, "BatchMVP")
plt.show()
