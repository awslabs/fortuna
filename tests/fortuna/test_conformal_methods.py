import unittest

import jax.numpy as jnp
import numpy as np

from fortuna.conformal.classification import (
    AdaptivePredictionConformalClassifier, SimplePredictionConformalClassifier, AdaptiveConformalClassifier,
    BatchMVPConformalClassifier
)
from fortuna.conformal.regression import (
    CVPlusConformalRegressor, EnbPI, JackknifeMinmaxConformalRegressor,
    JackknifePlusConformalRegressor,
    OneDimensionalUncertaintyConformalRegressor, QuantileConformalRegressor, AdaptiveConformalRegressor,
    BatchMVPConformalRegressor
)
from fortuna.data.loader import DataLoader, InputsLoader


np.random.rand(42)


class TestConformalMethods(unittest.TestCase):
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

        intervals = CVPlusConformalRegressor().conformal_interval(
            cross_val_outputs, cross_val_targets, cross_test_outputs, 0.05
        )
        assert intervals.ndim == 2
        assert intervals.shape[0] == m
        assert intervals.shape[1] == 2
        assert np.alltrue(intervals[:, 0] < intervals[:, 1])
        assert len(np.unique(intervals[:, 0])) > 1
        assert len(np.unique(intervals[:, 1])) > 1

    def test_jackknifeplus_conformal_regressor(self):
        n = 100
        m = 99
        loo_val_outputs = np.random.normal(size=(n, 1))
        loo_val_targets = np.random.normal(size=(n, 1))
        loo_test_outputs = np.random.normal(size=(n, m, 1))

        intervals = JackknifePlusConformalRegressor().conformal_interval(
            loo_val_outputs, loo_val_targets, loo_test_outputs, 0.05
        )
        assert intervals.ndim == 2
        assert intervals.shape[0] == m
        assert intervals.shape[1] == 2
        assert np.alltrue(intervals[:, 0] < intervals[:, 1])
        assert len(np.unique(intervals[:, 0])) > 1
        assert len(np.unique(intervals[:, 1])) > 1

    def test_jackknife_minmax_conformal_regressor(self):
        n = 100
        m = 99
        loo_val_outputs = np.random.normal(size=(n, 1))
        loo_val_targets = np.random.normal(size=(n, 1))
        loo_test_outputs = np.random.normal(size=(n, m, 1))

        intervals = JackknifeMinmaxConformalRegressor().conformal_interval(
            loo_val_outputs, loo_val_targets, loo_test_outputs, 0.05
        )
        assert intervals.ndim == 2
        assert intervals.shape[0] == m
        assert intervals.shape[1] == 2
        assert np.alltrue(intervals[:, 0] < intervals[:, 1])
        assert len(np.unique(intervals[:, 0])) > 1
        assert len(np.unique(intervals[:, 1])) > 1

    def test_enbpi(self):
        bs = [30, 10]
        t = 10
        t1 = 3
        error = 0.05

        for b in bs:
            # all without extra scalar dimension
            bootstrap_indices = np.random.choice(t, size=(b, t))
            bootstrap_train_preds = np.random.normal(size=(b, t))
            bootstrap_test_preds = np.random.normal(size=(b, t1))
            train_targets = np.random.normal(size=t)

            intervals = EnbPI().conformal_interval(
                bootstrap_indices=bootstrap_indices,
                bootstrap_train_preds=bootstrap_train_preds,
                bootstrap_test_preds=bootstrap_test_preds,
                train_targets=train_targets,
                error=error,
            )
            assert intervals.ndim == 2
            assert intervals.shape[0] == t1
            assert intervals.shape[1] == 2
            assert np.alltrue(intervals[:, 0] < intervals[:, 1])
            assert len(np.unique(intervals[:, 0])) > 1
            assert len(np.unique(intervals[:, 1])) > 1

            # all with extra scalar dimension
            bootstrap_train_preds = np.random.normal(size=(b, t, 1))
            bootstrap_test_preds = np.random.normal(size=(b, t1, 1))
            train_targets = np.random.normal(size=(t, 1))

            intervals = EnbPI().conformal_interval(
                bootstrap_indices=bootstrap_indices,
                bootstrap_train_preds=bootstrap_train_preds,
                bootstrap_test_preds=bootstrap_test_preds,
                train_targets=train_targets,
                error=error,
            )
            assert intervals.ndim == 2
            assert intervals.shape[0] == t1
            assert intervals.shape[1] == 2
            assert np.alltrue(intervals[:, 0] < intervals[:, 1])
            assert len(np.unique(intervals[:, 0])) > 1
            assert len(np.unique(intervals[:, 1])) > 1

            # predictions with and targets without extra scalar dimension
            train_targets = np.random.normal(size=t)

            intervals = EnbPI().conformal_interval(
                bootstrap_indices=bootstrap_indices,
                bootstrap_train_preds=bootstrap_train_preds,
                bootstrap_test_preds=bootstrap_test_preds,
                train_targets=train_targets,
                error=error,
            )
            assert intervals.ndim == 2
            assert intervals.shape[0] == t1
            assert intervals.shape[1] == 2
            assert np.alltrue(intervals[:, 0] < intervals[:, 1])
            assert len(np.unique(intervals[:, 0])) > 1
            assert len(np.unique(intervals[:, 1])) > 1

            # return also residuals
            train_targets = np.random.normal(size=t)

            intervals, residuals = EnbPI().conformal_interval(
                bootstrap_indices=bootstrap_indices,
                bootstrap_train_preds=bootstrap_train_preds,
                bootstrap_test_preds=bootstrap_test_preds,
                train_targets=train_targets,
                error=error,
                return_residuals=True,
            )

            assert intervals.ndim == 2
            assert intervals.shape[0] == t1
            assert intervals.shape[1] == 2
            assert np.alltrue(intervals[:, 0] < intervals[:, 1])
            assert len(np.unique(intervals[:, 0])) > 1
            assert len(np.unique(intervals[:, 1])) > 1
            assert residuals.shape == (t,)
            assert np.alltrue(residuals >= 0)

    def test_adaptive_conformal_regressor(self):
        acr = AdaptiveConformalRegressor(conformal_regressor=QuantileConformalRegressor())
        error = acr.update_error(
            conformal_interval=np.random.normal(size=2),
            error=0.01,
            target=np.array([1.2]),
            target_error=0.05
        )

        error = acr.update_error(
            conformal_interval=np.random.normal(size=2),
            error=0.01,
            target=np.array([1.2]),
            target_error=0.05,
            weights=np.array([0.1, 0.2, 0.3, 0.4]),
            were_in=np.array([1, 0, 1])
        )

    def test_adaptive_conformal_classification(self):
        acr = AdaptiveConformalClassifier(conformal_classifier=AdaptivePredictionConformalClassifier())
        error = acr.update_error(
            conformal_set=[2, 0],
            error=0.01,
            target=np.array([1]),
            target_error=0.05
        )

        error = acr.update_error(
            conformal_set=[2, 0],
            error=0.01,
            target=np.array([1]),
            target_error=0.05,
            weights=np.array([0.1, 0.2, 0.3, 0.4]),
            were_in=np.array([1, 0, 1])
        )

    def test_batchmvp_regressor(self):
        batchmvp = BatchMVPConformalRegressor(
            score_fn=lambda x, y: jnp.abs(y - x) / 15,
            group_fns=[lambda x: x > 0.1, lambda x: x < 0.2, lambda x: x > 0.3],
            bounds_fn=lambda x, t: (x - t, x + t)
        )
        val_data_loader = DataLoader.from_array_data(
            (np.random.normal(size=(50,)), np.random.normal(size=(50,))),
            batch_size=32,
        )
        test_inputs_loader = InputsLoader.from_array_inputs(
            np.random.normal(size=(150,)),
            batch_size=32,
        )

        intervals = batchmvp.conformal_interval(val_data_loader, test_inputs_loader)
        assert intervals.shape == (150, 2)

    def test_batchmvp_classifier(self):
        batchmvp = BatchMVPConformalClassifier(
            score_fn=lambda x, y: 1 - jnp.mean(x, 0)[y],
            group_fns=[lambda x: x[:, 0] > 0.1, lambda x: x[:, 0] < 0.2, lambda x: x[:, 0] > 0.3],
            n_classes=2
        )
        val_data_loader = DataLoader.from_array_data(
            (np.random.normal(size=(50, 1)), np.random.choice(2, 50)),
            batch_size=32,
        )
        test_inputs_loader = InputsLoader.from_array_inputs(
            np.random.normal(size=(150, 1)),
            batch_size=32,
        )

        sets = batchmvp.conformal_set(val_data_loader, test_inputs_loader)
        assert len(sets) == 150
