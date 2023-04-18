.. _model_outputs:

From model outputs
**********************************************************************************
Perhaps you have already trained your model,
and you are looking for calibrated uncertainty estimates starting from your model outputs.

If that is the case, you are in the right place.
In this scenario,
Fortuna can calibrate your model outputs,
estimate uncertainty,
compute metrics and obtain conformal prediction sets.
Your model may have been written and trained in any language;
Fortuna just needs model outputs and target variables in :code:`numpy.ndarray` or :code:`jax.numpy.ndarray` formats..

.. _model_outputs_classification:

Classification
================================
Let us show how to calibrate model outputs, estimate uncertainty, compute metrics and conformal sets in
classification.

Build a calibration classifier
------------------------------
First, let us build a calibration classifier.
This defines the :ref:`output calibrator <output_calibrator>` to attach to the model output,
and the final :ref:`probabilistic output layer <prob_output_layer>` used for calibration and to compute predictive
statistics.
The default output calibrator is temperature scaling, that is what we use in this example.

.. code-block:: python
    :caption: **References:** :class:`~fortuna.calibration.output_calib_model.classification.OutputCalibClassifier`

    from fortuna.calibration import OutputCalibClassifier
    calib_model = OutputCalibClassifier()

Calibrate the model outputs
-----------------------------
Let's calibrate the model outputs.
Fortuna needs an array of model outputs computed over some calibration inputs,
and a corresponding array of calibration target variables.
We denote these as :code:`calib_outputs` and :code:`calib_targets`, respectively.
You can configure the calibration process using a :ref:`calibration configuration <output_calib_model_config>` object.
In this example, we will stick with the default configuration options.

.. code-block:: python

    status = calib_model.calibrate(
        calib_outputs=calib_outputs,
        calib_targets=calib_targets
    )

Estimate statistics
-----------------------------
Given some test model outputs :code:`test_outputs`,
and potentially an array of test target variables :code:`test_targets`,
we are ready to estimate predictive statistics.
These include predictive mode, mean, log-pdf, variance, entropy, etc;
please consult the :ref:`predictive <calib_predictive_regression>` reference.

.. note::
    In classification, the predictive *mode* gives label predictions, i.e. the label
    predicted for a certain input, while the predictive *mean* gives probability predictions, i.e. the
    probability of each label.

.. code-block:: python
    :caption: **References:** :meth:`~fortuna.calibration.output_calib_model.predictive.classification.ClassificationPredictive.log_prob`, :meth:`~fortuna.calibration.output_calib_model.predictive.classification.ClassificationPredictive.mode`, :meth:`~fortuna.calibration.output_calib_model.predictive.classification.ClassificationPredictive.mean`

    test_logprob = calib_model.predictive.log_prob(
        outputs=test_outputs, targets=test_targets
    )
    test_modes = calib_model.predictive.mode(
        outputs=test_outputs
    )
    test_means = calib_model.predictive.mean(
        outputs=test_outputs
    )

Compute metrics
-----------------------------
Fortuna supports some classification metrics,
e.g. accuracy, expected calibration error and Brier score.
You are encouraged to bring in metrics from other frameworks and apply them on Fortuna's predictions,
as the latter are compatible with metrics operating on :code:`numpy.ndarray`.

.. code-block:: python
    :caption: **References:** :func:`~fortuna.metric.classification.accuracy`, :func:`~fortuna.metric.classification.expected_calibration_error`

    from fortuna.metric.classification import accuracy, expected_calibration_error
    acc = accuracy(
        preds=test_modes,
        targets=test_targets
    )
    ece = expected_calibration_error(
        preds=test_modes,
        probs=test_means,
        targets=test_targets
    )

Compute conformal sets
-----------------------------
Finally,
like in :ref:`conformal_classification_usage_mode`,
starting from predictive statistics you can compute conformal sets.
Again, we need model outputs and data for this purpose.
We denote :code:`val_outputs` to be validation model outputs,
and :code:`val_targets` to be the corresponding validation target variables.

.. code-block:: python
    :caption: **References:** :meth:`~fortuna.calibration.conformal.classification.adaptive_prediction.AdaptivePredictionConformalClassifier.conformal_set`

    from fortuna.calibration import AdaptivePredictionConformalClassifier
    val_means = calib_model.predictive.mean(
        outputs=val_outputs
    )
    conformal_sets = AdaptivePredictionConformalClassifier().conformal_set(
        val_probs=val_means,
        test_probs=test_means,
        val_targets=val_targets
    )

.. _model_outputs_regression:

Regression
================================
Similarly as in the :ref:`classification example <model_outputs_classification>`,
let us show how to calibrate model outputs, estimate uncertainty,
compute metrics and obtain conformal intervals in regression.

.. note::
    In regression,
    Fortuna requires model outputs to be concatenations of mean and log-variance models of a Gaussian likelihood function.
    Mathematically,
    suppose that :math:`\mu(\theta, x)` is the mean model,
    :math:`\sigma^2(\theta, x)` is a variance model,
    and :math:`N\Big(y|\mu(\theta, x), \sigma^2(\theta, x)\Big)` is likelihood function,
    where :math:`\theta` are model parameters,
    :math:`x` is an inputs variable and :math:`y` is an output variable.
    Then model outputs should be concatenations :math:`[\mu(\theta, x), \log\sigma^2(\theta, x)]`,
    for each input.

Build a calibration regressor
-----------------------------
First, let us build a calibration regressor.
This defines the :ref:`output calibrator <output_calibrator>` to attach to the model output,
and the final :ref:`probabilistic output layer <prob_output_layer>` used for calibration and to compute predictive
statistics.
The default output calibrator is temperature scaling, that is what we use in this example.

.. code-block:: python
    :caption: **References:** :class:`~fortuna.calibration.output_calib_model.regressor.OutputCalibRegressor`

    from fortuna.calibration import CalibRegression
    calib_model = OutputCalibRegressor()

Calibrate the model outputs
-----------------------------
Let's calibrate the model outputs.
Fortuna needs an array of model outputs computed over some calibration inputs,
and a corresponding array of calibration target variables.
We denote these as :code:`calib_outputs` and :code:`calib_targets`, respectively.
You can configure the calibration process using a :ref:`calibration configuration <output_calib_model_config>` object.
In this example, we will stick with the default configuration options.

.. code-block:: python

    status = calib_model.calibrate(
        calib_outputs=calib_outputs,
        calib_targets=calib_targets
    )

Estimate statistics
-----------------------------
Given some test model outputs :code:`test_outputs`,
and potentially an array of test target variables :code:`test_targets`,
we are ready to estimate predictive statistics.
These include predictive mode, mean, log-pdf, variance, entropy, etc;
please consult the :ref:`predictive <calib_predictive_classification>` reference.

.. note::
    In contrast with classification, in regression both the predictive *mean* and the predictive *mode* provide
    predictions for the target variables, and do not represent measures of uncertainty.

.. code-block:: python
    :caption: **References:** :meth:`~fortuna.calibration.output_calib_model.predictive.regression.RegressionPredictive.log_prob`, :meth:`~fortuna.calibration.output_calib_model.predictive.regression.RegressionPredictive.mean`, :meth:`~fortuna.calibration.output_calib_model.predictive.regression.RegressionPredictive.credible_interval`

    test_logprob = calib_model.predictive.log_prob(
        outputs=test_outputs, targets=test_targets
    )
    test_means = calib_model.predictive.mean(
        outputs=test_outputs
    )
    test_cred_intervals = calib_model.predictive.credible_interval(
        outputs=test_outputs
    )

Compute metrics
-----------------------------
Fortuna supports some regression metrics,
e.g. Root Mean-Squared Error (RMSE) and Prediction Interval Coverage Probability (PICP).
You are encouraged to bring in metrics from other frameworks and apply them on Fortuna's predictions,
as the latter are compatible with metrics operating on :code:`numpy.ndarray`.

.. code-block:: python
    :caption: **References:** :func:`~fortuna.metric.regression.root_mean_squared_error`, :func:`~fortuna.metric.regression.prediction_interval_coverage_probability`

    from fortuna.metric.regression import root_mean_squared_error, prediction_interval_coverage_probability
    rmse = root_mean_squared_error(
        preds=test_modes,
        targets=test_targets
    )
    picp = prediction_interval_coverage_probability(
        lower_bounds=test_cred_intervals[:, 0],
        upper_bounds=test_cred_intervals[:, 1],
        targets=test_targets
    )

Compute conformal intervals
-----------------------------
Finally,
like in :ref:`conformal_regression_example_credibility`,
starting from predictive statistics you can compute conformal intervals.
Again, we need model outputs and data for this purpose.
We denote :code:`val_outputs` to be validation model outputs,
and :code:`val_targets` to be the corresponding validation target variables.

.. code-block:: python
    :caption: **References:** :meth:`~fortuna.calibration.conformal.regression.quantile.QuantileConformalRegressor.conformal_interval`

    from fortuna.calibration import QuantileConformalRegressor
    val_cred_intervals = calib_model.predictive.credible_interval(
        outputs=val_outputs
    )
    conformal_intervals = QuantileConformalRegressor().conformal_intervals(
        val_lower_bounds=val_cred_intervals[:, 0],
        val_upper_bounds=valcalib_cred_intervals[:, 1],
        test_lower_bounds=test_cred_intervals[:, 0],
        test_upper_bounds=test_cred_intervals[:, 1],
        val_targets=val_targets
    )
