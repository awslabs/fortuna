import unittest

import jax.numpy as jnp
import numpy as np

from fortuna.conformal.classification import (
    AdaptivePredictionConformalClassifier, SimplePredictionConformalClassifier)
from fortuna.conformal.regression import (
    OneDimensionalUncertaintyConformalRegressor, QuantileConformalRegressor, CVPlusConformalRegressor,
    JackknifeMinmaxConformalRegressor, JackknifePlusConformalRegressor)


class TestConformals(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_prediction_conformal_classifier(self):
        val_probs = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        val_targets = np.array([1, 1, 0])
        test_probs = np.array([[0.5, 0.5], [0.8, 0.2]])

        conformal = SimplePredictionConformalClassifier()
        scores = conformal.score(val_probs, val_targets)
        assert scores.shape == (3,)
        quantile = conformal.quantile(val_probs, val_targets, 0.5, scores=scores)
        coverage_sets = conformal.conformal_set(
            val_probs=val_probs,
            test_probs=test_probs,
            val_targets=val_targets,
            quantile=quantile,
        )
        assert (
            (0 in coverage_sets[0])
            and (1 in coverage_sets[0])
            and (0 in coverage_sets[1])
        )

    def test_adaptive_prediction_conformal_classifier(self):
        val_probs = np.array([[0.1, 0.7, 0.2], [0.5, 0.4, 0.1], [0.15, 0.6, 0.35]])
        val_targets = np.array([1, 1, 2])
        test_probs = np.array([[0.2, 0.5, 0.3], [0.6, 0.01, 0.39]])

        conformal = AdaptivePredictionConformalClassifier()
        scores = conformal.score(val_probs, val_targets)
        assert jnp.allclose(scores, jnp.array([0.7, 0.9, 0.95]))
        quantile = conformal.quantile(val_probs, val_targets, 0.5, scores=scores)
        assert (quantile > 0.9) * (quantile < 0.95)
        coverage_sets = conformal.conformal_set(
            val_probs, test_probs, val_targets, quantile=quantile
        )
        assert np.allclose(coverage_sets[0], [1, 2, 0])
        assert np.allclose(coverage_sets[1], [0, 2])

    def test_quantile_conformal_regressor(self):
        n_val_inputs = 100
        n_test_inputs = 100
        val_lower_quantiles = np.random.normal(size=n_val_inputs)
        val_upper_quantiles = np.random.normal(size=n_val_inputs)
        test_lower_quantiles = np.random.normal(size=n_test_inputs)
        test_upper_quantiles = np.random.normal(size=n_test_inputs)
        val_targets = np.random.normal(size=(n_val_inputs, 1))

        conformal = QuantileConformalRegressor()
        scores = conformal.score(val_lower_quantiles, val_upper_quantiles, val_targets)
        assert scores.shape == (n_val_inputs,)
        quantile = conformal.quantile(
            val_lower_quantiles, val_upper_quantiles, val_targets, 0.05, scores=scores
        )
        assert jnp.min(scores) <= quantile <= jnp.max(scores)
        assert jnp.array([quantile]).shape == (1,)
        coverage_sets = conformal.conformal_interval(
            val_lower_quantiles,
            val_upper_quantiles,
            test_upper_quantiles,
            test_lower_quantiles,
            val_targets,
            0.05,
            quantile=quantile,
        )
        assert (
            coverage_sets[:, 0].shape == coverage_sets[:, 1].shape == (n_test_inputs,)
        )
        assert (coverage_sets[:, 0] < coverage_sets[:, 1]).all()

    def test_one_dimensional_uncertainty_conformal_regressor(self):
        n_val_inputs = 100
        n_test_inputs = 10
        val_preds = np.random.normal(size=n_val_inputs)[:, None]
        val_uncertainties = np.exp(np.random.normal(size=n_val_inputs))[:, None]
        test_preds = np.random.normal(size=n_test_inputs)[:, None]
        test_uncertainties = np.exp(np.random.normal(size=n_test_inputs))[:, None]
        val_targets = np.random.normal(size=(n_val_inputs, 1))

        conformal = OneDimensionalUncertaintyConformalRegressor()
        scores = conformal.score(val_preds, val_uncertainties, val_targets)
        assert (scores >= 0).all()
        assert scores.shape == (n_val_inputs,)
        quantile = conformal.quantile(
            val_preds=val_preds,
            val_uncertainties=val_uncertainties,
            val_targets=val_targets,
            error=0.05,
            scores=scores,
        )
        assert jnp.min(scores) <= quantile <= jnp.max(scores)
        assert jnp.array([quantile]).shape == (1,)
        coverage_sets = conformal.conformal_interval(
            val_preds=val_preds,
            val_uncertainties=val_uncertainties,
            test_preds=test_preds,
            test_uncertainties=test_uncertainties,
            val_targets=val_targets,
            error=0.05,
            quantile=quantile,
        )
        assert (
            coverage_sets[:, 0].shape == coverage_sets[:, 1].shape == (n_test_inputs,)
        )
        assert (coverage_sets[:, 0] < coverage_sets[:, 1]).all()

    def test_cvplus_conformal_regressor(self):
        k1, k2, k3 = 100, 101, 102
        m = 99
        cross_val_outputs = [
            np.random.normal(size=(k1, 1)),
            np.random.normal(size=(k2, 1)),
            np.random.normal(size=(k3, 1)),
        ]
        cross_val_targets = [
            np.random.normal(size=(k1, 1)),
            np.random.normal(size=(k2, 1)),
            np.random.normal(size=(k3, 1)),
        ]
        cross_test_outputs = [
            np.random.normal(size=(m, 1)),
            np.random.normal(size=(m, 1)),
            np.random.normal(size=(m, 1)),
        ]

        interval = CVPlusConformalRegressor().conformal_interval(cross_val_outputs, cross_val_targets, cross_test_outputs, 0.05)
        assert interval.ndim == 2
        assert interval.shape[0] == m
        assert interval.shape[1] == 2
        assert (interval[:, 0] < interval[:, 1]).all()
        assert len(np.unique(interval[:, 0])) > 1
        assert len(np.unique(interval[:, 1])) > 1

    def test_jackknifeplus_conformal_regressor(self):
        n = 100
        m = 99
        loo_val_outputs = np.random.normal(size=(n, 1))
        loo_val_targets = np.random.normal(size=(n, 1))
        loo_test_outputs = np.random.normal(size=(n, m, 1))

        interval = JackknifePlusConformalRegressor().conformal_interval(loo_val_outputs, loo_val_targets,
                                                                        loo_test_outputs, 0.05)
        assert interval.ndim == 2
        assert interval.shape[0] == m
        assert interval.shape[1] == 2
        assert (interval[:, 0] < interval[:, 1]).all()
        assert len(np.unique(interval[:, 0])) > 1
        assert len(np.unique(interval[:, 1])) > 1

    def test_jackknife_minmax_conformal_regressor(self):
        n = 100
        m = 99
        loo_val_outputs = np.random.normal(size=(n, 1))
        loo_val_targets = np.random.normal(size=(n, 1))
        loo_test_outputs = np.random.normal(size=(n, m, 1))

        interval = JackknifeMinmaxConformalRegressor().conformal_interval(loo_val_outputs, loo_val_targets,
                                                                          loo_test_outputs, 0.05)
        assert interval.ndim == 2
        assert interval.shape[0] == m
        assert interval.shape[1] == 2
        assert (interval[:, 0] < interval[:, 1]).all()
        assert len(np.unique(interval[:, 0])) > 1
        assert len(np.unique(interval[:, 1])) > 1