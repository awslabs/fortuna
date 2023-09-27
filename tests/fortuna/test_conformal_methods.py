import unittest

from jax import random
import jax.numpy as jnp
import numpy as np

from fortuna.conformal import (
    AdaptiveConformalClassifier,
    AdaptiveConformalRegressor,
    CVPlusSimplePredictionConformalClassifier,
    CVPlusAdaptivePredictionConformalClassifier,
    AdaptivePredictionConformalClassifier,
    BatchMVPConformalClassifier,
    BatchMVPConformalRegressor,
    BinaryClassificationMulticalibrator,
    CVPlusConformalRegressor,
    EnbPI,
    JackknifeMinmaxConformalRegressor,
    JackknifePlusConformalRegressor,
    MaxCoverageFixedPrecisionBinaryClassificationCalibrator,
    Multicalibrator,
    OneDimensionalUncertaintyConformalRegressor,
    OneShotBinaryClassificationMulticalibrator,
    OneShotMulticalibrator,
    OneShotTopLabelMulticalibrator,
    QuantileConformalRegressor,
    SimplePredictionConformalClassifier,
    TopLabelMulticalibrator,
)


class TestConformalMethods(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rng = np.random.default_rng(0)

    def test_prediction_conformal_classifier(self):
        val_probs = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        val_targets = np.array([1, 1, 0])
        test_probs = np.array([[0.5, 0.5], [0.8, 0.2]])

        conformal = SimplePredictionConformalClassifier()
        coverage_sets = conformal.conformal_set(
            val_probs=val_probs,
            test_probs=test_probs,
            val_targets=val_targets,
            error=0.05
        )
        assert (
            (0 in coverage_sets[0])
            and (1 in coverage_sets[0])
            and (0 in coverage_sets[1])
        )

    def test_simple_prediction_conformal_classifier(self):
        val_probs = np.array([[0.1, 0.7, 0.2], [0.5, 0.4, 0.1], [0.15, 0.6, 0.35]])
        val_targets = np.array([1, 1, 2])
        test_probs = np.array([[0.2, 0.5, 0.3], [0.6, 0.01, 0.39]])

        conformal = SimplePredictionConformalClassifier()
        coverage_sets = conformal.conformal_set(
            val_probs, val_targets, test_probs, error=0.05
        )
        assert len(coverage_sets) == len(test_probs)

        conformal = CVPlusSimplePredictionConformalClassifier()
        coverage_sets = conformal.conformal_set(
            [val_probs, val_probs], [val_targets, val_targets], [test_probs, test_probs], error=0.05
        )
        assert len(coverage_sets) == 2 * len(test_probs)

    def test_adaptive_prediction_conformal_classifier(self):
        val_probs = np.array([[0.1, 0.7, 0.2], [0.5, 0.4, 0.1], [0.15, 0.6, 0.35]])
        val_targets = np.array([1, 1, 2])
        test_probs = np.array([[0.2, 0.5, 0.3], [0.6, 0.01, 0.39]])

        conformal = AdaptivePredictionConformalClassifier()
        coverage_sets = conformal.conformal_set(
            val_probs, val_targets, test_probs, error=0.05
        )
        assert len(coverage_sets) == len(test_probs)

        conformal = CVPlusAdaptivePredictionConformalClassifier()
        coverage_sets = conformal.conformal_set(
            [val_probs, val_probs], [val_targets, val_targets], [test_probs, test_probs], error=0.05
        )
        assert len(coverage_sets) == 2 * len(test_probs)

    def test_quantile_conformal_regressor(self):
        n_val_inputs = 100
        n_test_inputs = 100
        val_lower_quantiles = self._rng.normal(size=n_val_inputs)
        val_upper_quantiles = self._rng.normal(size=n_val_inputs)
        test_lower_quantiles = self._rng.normal(size=n_test_inputs)
        test_upper_quantiles = self._rng.normal(size=n_test_inputs)
        val_targets = self._rng.normal(size=(n_val_inputs, 1))

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
        val_preds = self._rng.normal(size=n_val_inputs)[:, None]
        val_uncertainties = np.exp(self._rng.normal(size=n_val_inputs))[:, None]
        test_preds = self._rng.normal(size=n_test_inputs)[:, None]
        test_uncertainties = np.exp(self._rng.normal(size=n_test_inputs))[:, None]
        val_targets = self._rng.normal(size=(n_val_inputs, 1))

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
            self._rng.normal(size=(k1, 1)),
            self._rng.normal(size=(k2, 1)),
            self._rng.normal(size=(k3, 1)),
        ]
        cross_val_targets = [
            self._rng.normal(size=(k1, 1)),
            self._rng.normal(size=(k2, 1)),
            self._rng.normal(size=(k3, 1)),
        ]
        cross_test_outputs = [
            self._rng.normal(size=(m, 1)),
            self._rng.normal(size=(m, 1)),
            self._rng.normal(size=(m, 1)),
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
        loo_val_outputs = self._rng.normal(size=(n, 1))
        loo_val_targets = self._rng.normal(size=(n, 1))
        loo_test_outputs = self._rng.normal(size=(n, m, 1))

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
        loo_val_outputs = self._rng.normal(size=(n, 1))
        loo_val_targets = self._rng.normal(size=(n, 1))
        loo_test_outputs = self._rng.normal(size=(n, m, 1))

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
            bootstrap_indices = self._rng.choice(t, size=(b, t))
            bootstrap_train_preds = self._rng.normal(size=(b, t))
            bootstrap_test_preds = self._rng.normal(size=(b, t1))
            train_targets = self._rng.normal(size=t)

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
            bootstrap_train_preds = self._rng.normal(size=(b, t, 1))
            bootstrap_test_preds = self._rng.normal(size=(b, t1, 1))
            train_targets = self._rng.normal(size=(t, 1))

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
            train_targets = self._rng.normal(size=t)

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
            train_targets = self._rng.normal(size=t)

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
        acr = AdaptiveConformalRegressor(
            conformal_regressor=QuantileConformalRegressor()
        )
        error = acr.update_error(
            conformal_interval=self._rng.normal(size=2),
            error=0.01,
            target=np.array([1.2]),
            target_error=0.05,
        )

        error = acr.update_error(
            conformal_interval=self._rng.normal(size=2),
            error=0.01,
            target=np.array([1.2]),
            target_error=0.05,
            weights=np.array([0.1, 0.2, 0.3, 0.4]),
            were_in=np.array([1, 0, 1]),
        )

    def test_adaptive_conformal_classification(self):
        acr = AdaptiveConformalClassifier(
            conformal_classifier=AdaptivePredictionConformalClassifier()
        )
        error = acr.update_error(
            conformal_set=[2, 0], error=0.01, target=np.array([1]), target_error=0.05
        )

        error = acr.update_error(
            conformal_set=[2, 0],
            error=0.01,
            target=np.array([1]),
            target_error=0.05,
            weights=np.array([0.1, 0.2, 0.3, 0.4]),
            were_in=np.array([1, 0, 1]),
        )

    def test_batchmvp_regressor(self):
        size = 10
        test_size = 20
        scores = random.uniform(random.PRNGKey(0), shape=(size,))
        groups = random.choice(random.PRNGKey(0), 2, shape=(size, 3)).astype("bool")
        values = jnp.zeros(size)
        test_scores = random.uniform(random.PRNGKey(0), shape=(test_size,))
        test_groups = random.choice(random.PRNGKey(1), 2, shape=(test_size, 3)).astype(
            "bool"
        )
        batchmvp = BatchMVPConformalRegressor()
        status = batchmvp.calibrate(scores=scores, n_rounds=3, n_buckets=4)
        status = batchmvp.calibrate(
            scores=scores, groups=groups, thresholds=values, n_rounds=3, n_buckets=4
        )
        test_values, status = batchmvp.calibrate(
            scores=scores,
            groups=groups,
            test_groups=test_groups,
            n_rounds=3,
            n_buckets=4,
        )
        with self.assertRaises(ValueError):
            test_values, status = batchmvp.calibrate(
                scores=scores,
                groups=groups,
                thresholds=values,
                test_groups=test_groups,
                n_rounds=3,
                n_buckets=4,
            )
        with self.assertRaises(ValueError):
            test_values, status = batchmvp.calibrate(
                scores=scores,
                groups=groups,
                thresholds=values,
                test_thresholds=test_values,
                n_rounds=3,
                n_buckets=4,
            )
        with self.assertRaises(ValueError):
            test_thresholds, status = batchmvp.calibrate(
                scores=scores,
                groups=groups,
                test_groups=test_groups,
                test_thresholds=test_values,
                n_rounds=3,
                n_buckets=4,
            )
        status = batchmvp.calibrate(
            scores=scores, groups=groups, n_rounds=3, n_buckets=4
        )
        test_values = batchmvp.apply_patches(test_groups)
        test_values = batchmvp.apply_patches(test_groups, test_values)
        error = batchmvp.calibration_error(
            scores=test_scores, groups=test_groups, thresholds=test_values
        )
        status = batchmvp.calibrate(
            scores=np.array(scores),
            thresholds=np.array(values),
            groups=np.array(groups),
            test_groups=np.array(test_groups),
            test_thresholds=np.array(test_values),
            n_rounds=3,
            n_buckets=4,
        )

    def test_batchmvp_classifier(self):
        size = 10
        test_size = 20
        scores = random.uniform(random.PRNGKey(0), shape=(size,))
        groups = random.choice(random.PRNGKey(0), 2, shape=(size, 3)).astype("bool")
        values = jnp.zeros(size)
        test_scores = random.uniform(random.PRNGKey(0), shape=(test_size,))
        test_groups = random.choice(random.PRNGKey(1), 2, shape=(test_size, 3)).astype(
            "bool"
        )
        batchmvp = BatchMVPConformalClassifier()
        status = batchmvp.calibrate(scores=scores, n_rounds=3, n_buckets=4)
        status = batchmvp.calibrate(
            scores=scores, groups=groups, thresholds=values, n_rounds=3, n_buckets=4
        )
        test_values, status = batchmvp.calibrate(
            scores=scores,
            groups=groups,
            test_groups=test_groups,
            n_rounds=3,
            n_buckets=4,
        )
        with self.assertRaises(ValueError):
            test_values, status = batchmvp.calibrate(
                scores=scores,
                groups=groups,
                thresholds=values,
                test_groups=test_groups,
                n_rounds=3,
                n_buckets=4,
            )
        with self.assertRaises(ValueError):
            test_values, status = batchmvp.calibrate(
                scores=scores,
                groups=groups,
                thresholds=values,
                test_thresholds=test_values,
                n_rounds=3,
                n_buckets=4,
            )
        with self.assertRaises(ValueError):
            test_values, status = batchmvp.calibrate(
                scores=scores,
                groups=groups,
                test_groups=test_groups,
                test_thresholds=test_values,
                n_rounds=3,
                n_buckets=4,
            )
        status = batchmvp.calibrate(
            scores=scores, groups=groups, n_rounds=3, n_buckets=4
        )
        test_values = batchmvp.apply_patches(test_groups)
        test_values = batchmvp.apply_patches(test_groups, test_values)
        error = batchmvp.calibration_error(
            scores=test_scores, groups=test_groups, thresholds=test_values
        )
        status = batchmvp.calibrate(
            scores=np.array(scores),
            thresholds=np.array(values),
            groups=np.array(groups),
            test_groups=np.array(test_groups),
            test_thresholds=np.array(test_values),
            n_rounds=3,
            n_buckets=4,
        )

        sets = batchmvp.conformal_set(
            class_scores=jnp.stack((test_scores, test_scores), axis=1),
            thresholds=test_values,
        )
        assert len(sets) == test_size

    def test_multicalibrator(self):
        size = 10
        test_size = 20
        scores = random.uniform(random.PRNGKey(0), shape=(size,))
        groups = random.choice(random.PRNGKey(0), 2, shape=(size, 3)).astype("bool")
        values = jnp.zeros(size)
        test_scores = random.uniform(random.PRNGKey(0), shape=(test_size,))
        test_groups = random.choice(random.PRNGKey(1), 2, shape=(test_size, 3)).astype(
            "bool"
        )
        mc = Multicalibrator()
        status = mc.calibrate(scores=scores, n_rounds=3, n_buckets=4)
        status = mc.calibrate(
            scores=scores, groups=groups, values=values, n_rounds=3, n_buckets=4
        )
        test_values, status = mc.calibrate(
            scores=scores,
            groups=groups,
            test_groups=test_groups,
            n_rounds=3,
            n_buckets=4,
        )
        with self.assertRaises(ValueError):
            test_values, status = mc.calibrate(
                scores=scores,
                groups=groups,
                values=values,
                test_groups=test_groups,
                n_rounds=3,
                n_buckets=4,
            )
        with self.assertRaises(ValueError):
            test_values, status = mc.calibrate(
                scores=scores,
                groups=groups,
                values=values,
                test_values=test_values,
                n_rounds=3,
                n_buckets=4,
            )
        with self.assertRaises(ValueError):
            test_values, status = mc.calibrate(
                scores=scores,
                groups=groups,
                test_groups=test_groups,
                test_values=test_values,
                n_rounds=3,
                n_buckets=4,
            )
        status = mc.calibrate(scores=scores, groups=groups, n_rounds=3, n_buckets=4)
        test_values = mc.apply_patches(test_groups)
        test_values = mc.apply_patches(test_groups, test_values)
        error = mc.calibration_error(
            scores=test_scores, groups=test_groups, values=test_values
        )
        status = mc.calibrate(
            scores=np.array(scores),
            values=np.array(values),
            groups=np.array(groups),
            test_groups=np.array(test_groups),
            test_values=np.array(test_values),
            n_rounds=3,
            n_buckets=4,
        )

    def test_binary_multicalibrator(self):
        size = 10
        test_size = 20
        scores = random.choice(random.PRNGKey(1), 2, shape=(size,)).astype("int")
        groups = random.choice(random.PRNGKey(0), 2, shape=(size, 3)).astype("bool")
        values = jnp.zeros(size)
        test_scores = random.choice(random.PRNGKey(1), 2, shape=(test_size,)).astype(
            "int"
        )
        test_groups = random.choice(random.PRNGKey(1), 2, shape=(test_size, 3)).astype(
            "bool"
        )
        mc = BinaryClassificationMulticalibrator()
        status = mc.calibrate(targets=scores, n_rounds=3, n_buckets=4)
        status = mc.calibrate(
            targets=scores, groups=groups, probs=values, n_rounds=3, n_buckets=4
        )
        test_values, status = mc.calibrate(
            targets=scores,
            groups=groups,
            test_groups=test_groups,
            n_rounds=3,
            n_buckets=4,
        )
        with self.assertRaises(ValueError):
            test_values, status = mc.calibrate(
                targets=scores,
                groups=groups,
                probs=values,
                test_groups=test_groups,
                n_rounds=3,
                n_buckets=4,
            )
        with self.assertRaises(ValueError):
            test_values, status = mc.calibrate(
                targets=scores,
                groups=groups,
                probs=values,
                test_probs=test_values,
                n_rounds=3,
                n_buckets=4,
            )
        with self.assertRaises(ValueError):
            test_values, status = mc.calibrate(
                targets=scores,
                groups=groups,
                test_groups=test_groups,
                test_probs=test_values,
                n_rounds=3,
                n_buckets=4,
            )
        status = mc.calibrate(targets=scores, groups=groups, n_rounds=3, n_buckets=4)
        test_values = mc.apply_patches(test_groups)
        test_values = mc.apply_patches(test_groups, test_values)
        error = mc.calibration_error(
            targets=test_scores, groups=test_groups, probs=test_values
        )
        status = mc.calibrate(
            targets=np.array(scores),
            probs=np.array(values),
            groups=np.array(groups),
            test_groups=np.array(test_groups),
            test_probs=np.array(test_values),
            n_rounds=3,
            n_buckets=4,
        )

    def test_top_label_classification_multicalibrator(self):
        size = 30
        test_size = 20
        n_classes = 3
        n_groups = 2
        scores = random.choice(random.PRNGKey(0), n_classes, shape=(size,)).astype(
            "int"
        )
        groups = random.choice(random.PRNGKey(0), 2, shape=(size, n_groups)).astype(
            "bool"
        )
        values = jnp.ones((size, n_classes))
        test_values = jnp.ones((test_size, n_classes))
        test_scores = random.choice(
            random.PRNGKey(1), n_classes, shape=(test_size,)
        ).astype("int")
        test_groups = random.choice(
            random.PRNGKey(1), 2, shape=(test_size, n_groups)
        ).astype("bool")
        mc = TopLabelMulticalibrator(n_classes=n_classes)
        status = mc.calibrate(targets=scores, n_rounds=3, n_buckets=4)
        status = mc.calibrate(
            targets=scores, groups=groups, probs=values, n_rounds=3, n_buckets=4
        )
        calib_test_values, status = mc.calibrate(
            targets=scores,
            groups=groups,
            test_groups=test_groups,
            n_rounds=3,
            n_buckets=10,
        )
        with self.assertRaises(ValueError):
            calib_test_values, status = mc.calibrate(
                targets=scores,
                groups=groups,
                probs=values,
                test_groups=test_groups,
                n_rounds=3,
                n_buckets=4,
            )
        with self.assertRaises(ValueError):
            calib_test_values, status = mc.calibrate(
                targets=scores,
                groups=groups,
                probs=values,
                test_probs=test_values,
                n_rounds=3,
                n_buckets=4,
            )
        with self.assertRaises(ValueError):
            calib_test_values, status = mc.calibrate(
                targets=scores,
                groups=groups,
                test_groups=test_groups,
                test_probs=test_values,
                n_rounds=3,
                n_buckets=4,
            )
        status = mc.calibrate(targets=scores, groups=groups, n_rounds=3, n_buckets=4)
        calib_test_values = mc.apply_patches(test_groups)
        calib_test_values = mc.apply_patches(test_groups, test_values)
        error = mc.calibration_error(
            targets=test_scores, groups=test_groups, probs=test_values
        )
        status = mc.calibrate(
            targets=np.array(scores),
            probs=np.array(values),
            groups=np.array(groups),
            test_groups=np.array(test_groups),
            test_probs=np.array(test_values) if test_values is not None else None,
            n_rounds=3,
            n_buckets=4,
        )

    def test_one_shot_multicalibrator(self):
        size = 10
        test_size = 20
        scores = random.uniform(random.PRNGKey(0), shape=(size,))
        values = jnp.zeros(size)
        test_scores = random.uniform(random.PRNGKey(0), shape=(test_size,))
        test_values = jnp.zeros(test_size)
        mc = OneShotMulticalibrator()
        mc.calibrate(
            scores=scores,
            values=values,
            n_buckets=4,
        )
        test_values = mc.calibrate(
            scores=scores,
            values=values,
            test_values=test_values,
            n_buckets=4,
        )
        with self.assertRaises(ValueError):
            test_values = mc.calibrate(
                scores=scores,
                test_values=test_values,
                n_buckets=4,
            )
        test_values = mc.apply_patches(test_values)
        error = mc.mean_squared_error(
            values=test_values,
            scores=test_scores,
        )
        mc.calibrate(
            scores=np.array(scores),
            values=np.array(values),
            test_values=np.array(test_values),
            n_buckets=4,
        )

    def test_binary_one_shot_multicalibrator(self):
        size = 10
        test_size = 20
        scores = random.choice(random.PRNGKey(0), 2, shape=(size,)).astype("int")
        values = jnp.zeros(size)
        test_scores = random.choice(random.PRNGKey(1), 2, shape=(test_size,)).astype(
            "int"
        )
        test_values = jnp.zeros(test_size)
        mc = OneShotBinaryClassificationMulticalibrator()
        mc.calibrate(
            targets=scores,
            probs=values,
            n_buckets=4,
        )
        test_values = mc.calibrate(
            targets=scores,
            probs=values,
            test_probs=test_values,
            n_buckets=4,
        )
        with self.assertRaises(ValueError):
            test_values = mc.calibrate(
                targets=scores,
                test_probs=test_values,
                n_buckets=4,
            )
        test_values = mc.apply_patches(test_values)
        error = mc.mean_squared_error(
            probs=test_values,
            targets=test_scores,
        )
        mc.calibrate(
            targets=np.array(scores),
            probs=np.array(values),
            test_probs=np.array(test_values),
            n_buckets=4,
        )

    def test_one_shot_top_label_multicalibrator(self):
        size = 30
        test_size = 20
        n_classes = 3
        scores = random.choice(random.PRNGKey(0), n_classes, shape=(size,)).astype(
            "int"
        )
        values = jnp.zeros((size, n_classes))
        test_values = jnp.zeros((test_size, n_classes))
        test_scores = random.choice(
            random.PRNGKey(1), n_classes, shape=(test_size,)
        ).astype("int")
        mc = OneShotTopLabelMulticalibrator(n_classes=n_classes)
        mc.calibrate(
            targets=scores,
            probs=values,
            n_buckets=4,
        )
        test_values = mc.calibrate(
            targets=scores,
            probs=values,
            test_probs=test_values,
            n_buckets=4,
        )
        with self.assertRaises(ValueError):
            test_values = mc.calibrate(
                targets=scores,
                test_probs=test_values,
                n_buckets=4,
            )
        test_values = mc.apply_patches(test_values)
        error = mc.mean_squared_error(
            probs=test_values,
            targets=test_scores,
        )
        mc.calibrate(
            targets=np.array(scores),
            probs=np.array(values),
            test_probs=np.array(test_values),
            n_buckets=4,
        )

    def test_max_coverage_fixed_precision_binary_classification_calibrator(self):
        size = 30
        test_size = 20
        targets = random.choice(random.PRNGKey(0), 2, shape=(size,)).astype("int")
        probs = jnp.zeros(size)
        test_probs = jnp.zeros(test_size)
        calib = MaxCoverageFixedPrecisionBinaryClassificationCalibrator()
        calib.calibrate(
            targets=targets,
            probs=probs,
            true_positive_precision_threshold=0.99,
            false_negative_precision_threshold=0.99,
        )
        test_values = calib.calibrate(
            targets=targets,
            probs=probs,
            test_probs=test_probs,
            true_positive_precision_threshold=0.99,
            false_negative_precision_threshold=0.99,
        )
