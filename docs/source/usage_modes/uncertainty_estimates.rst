.. _conformal:

From uncertainty estimates
**********************************************************************************
Fortuna provides some `conformal prediction methods <https://arxiv.org/abs/2107.07511>`__.
These start from uncertainty estimates and some calibration data, and construct *conformal predictions sets*,
i.e. rigorous sets of predictions that are probable above a certain threshold.

.. caution::
    If you provide bad uncertainty estimates, the conformal prediction sets may be very large and uninformative.
    If possible, please consider letting Fortuna estimating uncertainty - see :ref:`flax_models`.

.. _conformal_classification:

Classification
==============
In classification, conformal sets are collections of labels that are probable above a certain
thresholds. In the following example, we show how you can use Fortuna to construct them.

.. _conformal_classification_example:

Classification example: conformal intervals from probability predictions
-------------------------------------------------------------------------------------------------------
We assume you have trained a model and, for each input,
you have a way of estimating the probability of each label.
We call :code:`val_probs` and :code:`test_probs` these probabilities computed on some validation and test data sets,
respectively.
We further require the array of validation target variables :code:`val_targets` corresponding to the
validation probabilities. Then the following code provides a 95% conformal set for each test input.
Please check :class:`~fortuna.conformer.classification.AdaptivePredictionConformalClassifier` for reference.

.. code-block:: python
    :caption: **References:** :meth:`~fortuna.conformer.classification.AdaptivePredictionConformalClassifier.conformal_set`

    from fortuna.conformer.classification import AdaptivePredictionConformalClassifier
    conformal_sets = AdaptivePredictionConformalClassifier().conformal_set(
        val_probs=val_probs,
        test_probs=test_probs,
        val_targets=val_targets
    )

You should usually expect your test predictions to be included in the conformal sets, as they contain the most probable
labels according to your validation data and validation probability estimates.
You should also expect, overall, smaller sets for well-classified inputs and larger sets for misclassified ones,
as the latter are likely to be more uncertain. Notice that if your estimated probabilities are very uninformative or
highly wrong, the conformal sets might include almost all labels, signalling high uncertainty in the model.

.. _conformal_regression:

Regression
==========
For regression tasks with scalar target variables, Fortuna offers conformal methods that construct
conformal intervals. You can think of these as calibrated versions of confidence or credibility
intervals. In the following examples, we show how you can use Fortuna to construct them.

For both examples, you should usually expect your test predictions to be included in the
conformal intervals,
as they contain the range of most probable predictions according to your validation data and validation intervals.
You should also expect, overall, smaller ranges for well-classified inputs and larger sets for misclassified ones,
as the latter are likely to be more uncertain. Notice that if your intervals are uninformative or wrong,
the conformal intervals might be large.

.. _conformal_regression_example_credibility:

Conformal intervals from confidence or credibility intervals
------------------------------------------------------------------------------------
For this example,
we assume you have trained a model, and a way of estimating a confidence or credible interval for
each of your predictions.
We call :code:`val_lower_bounds`, :code:`val_upper_bounds`, :code:`test_lower_bounds` and
:code:`test_upper_bounds` the lower and upper bounds of these intervals computed on some validation and
test data sets. We suppose these have an error level given by :code:`error`; for example, 95% intervals
have `error=0.05`.
We further require the array of validation target variables :code:`val_targets` corresponding to the
validation intervals.
Then the following code provides a conformal interval with level `error` of error for each test input.
Please see :class:`~fortuna.conformer.regression.QuantileConformalRegressor` for reference.

.. code-block:: python
    :caption: **References:** :meth:`~fortuna.conformer.regression.QuantileConformalRegressor.conformal_interval`

    from fortuna.conformer.regression import QuantileConformalRegressor
    conformal_intervals = QuantileConformalRegressor().conformal_interval(
        val_lower_bounds=val_lower_bounds,
        val_upper_bounds=val_upper_bounds,
        test_lower_bounds=test_lower_bounds,
        test_upper_bounds=test_upper_bounds,
        val_targets=val_targets,
        error=error
    )

.. _conformal_regression_example_uncertainty:

Conformal intervals from confidence or scalar uncertainty estimates
-------------------------------------------------------------------------------------------------------
For this example,
we assume you have trained a model, you have a way of making predictions and estimating a scalar measure of
predictive uncertainty,
e.g. the standard deviation.
We call :code:`val_preds`, :code:`val_uncertainties`, :code:`test_preds` and
:code:`test_uncertainties` the predictions and uncertainties computed on some validation and test data sets.
We further require the array of validation target variables :code:`val_targets` corresponding to the
validation predictions.
Then the following code provides a 95% conformal interval for each test input.
Please see :class:`~fortuna.conformer.regression.OneDimensionalUncertaintyConformalRegressor` for reference.

.. code-block:: python
    :caption: **References:** :meth:`~fortuna.conformer.regression.OneDimensionalUncertaintyConformalRegressor.conformal_interval`

    from fortuna.conformer.regression import OneDimensionalUncertaintyConformalRegressor
    conformal_intervals = OneDimensionalUncertaintyConformalRegressor().conformal_interval(
        val_preds=val_preds,
        val_uncertainties=val_uncertainties,
        test_preds=test_preds,
        test_uncertainties=test_uncertainties,
        val_targets=val_targets
    )

