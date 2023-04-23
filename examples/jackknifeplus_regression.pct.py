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
# # Jackknife+, Jackknife-minmax and CV+

# %% [markdown]
# In this notebook we compare `Jackknife+`, `Jackknife-minmax` and `CV+` from [Barber et al. 2021](https://projecteuclid.org/journals/annals-of-statistics/volume-49/issue-1/Predictive-inference-with-the-jackknife/10.1214/20-AOS1965.full).


# %% [markdown]
# ## Generate regression data

# %% [markdown]
# We generate an arbitrary regression data set with scalar target variables. We split it into train and test set.

# %%
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X, y = make_regression(n_samples=2000, n_features=3, n_targets=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0, test_size=1000
)

# %% [markdown]
# We arbitrarily decide to adopt a gradient boosting method for regression.

# %%
from sklearn.ensemble import GradientBoostingRegressor

# %% [markdown]
# We decide for an arbitrary desired coverage of 95%.

# %%
from fortuna.metric.regression import prediction_interval_coverage_probability

error = 0.05

# %% [markdown]
# ## CV+

# %% [markdown]
# First, we train the model using a K-fold cross validation procedure.

# %%
from sklearn.model_selection import KFold

cross_val_outputs, cross_val_targets, cross_test_outputs = [], [], []
n_splits = 5
for i, idx in enumerate(KFold(n_splits=n_splits).split(X_train)):
    print(f"Split #{i + 1} out of {n_splits}.", end="\r")
    model = GradientBoostingRegressor()
    model.fit(X_train[idx[0]], y_train[idx[0]])
    cross_val_outputs.append(model.predict(X_train[idx[1]]))
    cross_val_targets.append(y_train[idx[1]])
    cross_test_outputs.append(model.predict(X_test))

# %% [markdown]
# Given the model outputs, we compute conformal intervals obtained using CV+.

# %%
from fortuna.conformal import CVPlusConformalRegressor

cvplus_interval = CVPlusConformalRegressor().conformal_interval(
    cross_val_outputs=cross_val_outputs,
    cross_val_targets=cross_val_targets,
    cross_test_outputs=cross_test_outputs,
    error=error,
)
cvplus_coverage = prediction_interval_coverage_probability(
    cvplus_interval[:, 0], cvplus_interval[:, 1], y_test
)

# %% [markdown]
# # Jackknife+ and jackknife-minmax

# %% [markdown]
# We now train the model with a leave-one-out procedure.

# %%
from sklearn.model_selection import LeaveOneOut
import jax.numpy as jnp

loo_val_outputs, loo_val_targets, loo_test_outputs = [], [], []
c = 0
for i, idx in enumerate(LeaveOneOut().split(X_train)):
    if c >= 30:
        break
    print(f"Split #{i + 1} out of {X_train.shape[0]}.", end="\r")
    model = GradientBoostingRegressor()
    model.fit(X_train[idx[0]], y_train[idx[0]])
    loo_val_outputs.append(model.predict(X_train[idx[1]]))
    loo_val_targets.append(y_train[idx[1]])
    loo_test_outputs.append(model.predict(X_test))
    c += 1

loo_val_outputs = jnp.array(loo_val_outputs)
loo_val_targets = jnp.array(loo_val_targets)
loo_test_outputs = jnp.array(loo_test_outputs)

# %% [markdown]
# Given the model outputs, we compute conformal intervals obtained using jackknife+ and jackknife-minmax.

# %%
from fortuna.conformal import (
    JackknifePlusConformalRegressor,
    JackknifeMinmaxConformalRegressor,
)

jkplus_interval = JackknifePlusConformalRegressor().conformal_interval(
    loo_val_outputs=loo_val_outputs,
    loo_val_targets=loo_val_targets,
    loo_test_outputs=loo_test_outputs,
    error=error,
)
jkplus_coverage = prediction_interval_coverage_probability(
    jkplus_interval[:, 0], jkplus_interval[:, 1], y_test
)

jkmm_interval = JackknifeMinmaxConformalRegressor().conformal_interval(
    loo_val_outputs=loo_val_outputs,
    loo_val_targets=loo_val_targets,
    loo_test_outputs=loo_test_outputs,
    error=error,
)
jkmm_coverage = prediction_interval_coverage_probability(
    jkmm_interval[:, 0], jkmm_interval[:, 1], y_test
)

# %% [markdown]
# ## Coverage results

# %%
print(f"Desired coverage: {1 - error}.")
print(f"CV+ empirical coverage: {cvplus_coverage}.")
print(f"jackknife+ empirical coverage: {jkplus_coverage}.")
print(f"jackknife-minmax empirical coverage: {jkmm_coverage}.")

# %% [markdown]
# Compared to CV+, where we trained the model with a 5-fold cross validation, jackknife+ and jackknife-minmax required significantly higher computational time because of the leave-one-out procedure over the whole training data set. One may significantly reduce this cost by performing leave-one-out only on a subset of the training data.

# %%
