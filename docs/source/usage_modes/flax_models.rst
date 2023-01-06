.. _flax_models:

From Flax models
**********************************************************************************
You can use Fortuna to fit the posterior distribution via scalable Bayesian inference methods,
calibrate model outputs,
make uncertainty estimates,
compute metrics,
and obtain conformal prediction sets.
The combination of all these methodologies gives you the highest chances to obtain
reliable calibrated uncertainty estimates.

In order to do so, you need to build a deep-learning model in `Flax <https://flax.readthedocs.io/>`__
(powered by `JAX <https://jax.readthedocs.io/en/latest/>`__),
or select a pre-built one from the models already implemented within Fortuna.
It is also up to you to provide some data,
either in array format or as a data loader.
You can use :class:`~fortuna.data.loader.DataLoader` to easily convert these into something Fortuna can digest.

That's it. These are the minimal requirements to let Fortuna train the model and get calibrated
predictions.

Of course, if you are familiar with machine learning and uncertainty estimation, you are welcome to
build a probabilistic model and configure training and calibration methods in the way that best serves
your need. The better your choices, the better your results.

.. _flax_models_classification:

Classification
================================
Let us show how to train a model, obtain predictions, compute metrics and conformal sets in
classification.

Provide data loaders
-----------------------------
First, you are required to provide some training data :code:`train_data`. This must be
converted into a data loader compatible with Fortuna. In this example, we assume your data is a
tuple of arrays, respectively containing input and target variables. Several other data formats are
supported, e.g. PyTorch data loaders, TensorFlow data loader, etc. Please look into :class:`~fortuna.data.loader.DataLoader` to
explore the supported conversion options.

.. code-block:: python
    :caption: **References:** :meth:`~fortuna.data.loader.DataLoader.from_array_data`

    from fortuna import DataLoader
    train_data_loader = DataLoader.from_array_data(
        train_data,
        batch_size=128,
        shuffle=True,
        prefetch=True
    )

Choose or build a model
-----------------------------
Second, you must provide a deep learning model from input variables to the logits of a softmax
function. The output dimension, i.e. the number of logits, must
corresponds to the number of different labels in the data.
For this, you can either choose one of Fortuna's pre-built models, or build your own in
`Flax <https://flax.readthedocs.io/>`__.
In this code example, we take a pre-built Multi-Layer Perceptron (MLP),
and denote the output dimension by :code:`output_dim`.

.. code-block:: python
    :caption: **References:** :class:`~fortuna.prob_model.model.mlp.MLP`

    from fortuna.model import MLP
    model = MLP(
        output_dim=output_dim
    )

Build a probabilistic model
-----------------------------
Fortuna now wraps the model into a probabilistic classifier, your interface object. You may configure
this by choosing a prior distribution, a posterior approximation method, an output calibration method
and a random seed; please consult the :ref:`probabilistic classifier <prob_classifier>` reference.
However, if you are not particularly familiar with these concepts, just go with the default options,
like we do in this example.

.. code-block:: python
    :caption: **References:** :class:`~fortuna.prob_model.classification.ProbClassifier`

    from fortuna import ProbClassifier
    prob_model = ProbClassifier(
        model=model
    )

Train the probabilistic model
-----------------------------
Let's train the probabilistic model now. Training includes two parts: fitting the posterior
distribution, and post-processing calibration. However, calibration is performed only if a calibration
data loader is provided. As for the training data, for this example we assume you have calibration data
:code:`calib_data` formatted as a tuple of arrays, and let Fortuna transform it into a data loader. The
training returns a status object, describing the progress of the training process.

You are encouraged to configure fitting and calibration as you please. You can specify the optimization
process, saving and restoring checkpoints, monitoring the training and enable early stopping, and
select your computation device; please consult the :ref:`fit configuration <fit_config>` and
:ref:`calibration configuration <prob_model_calib_config>` references. In this example, we will stick with the
default configuration options.

.. code-block:: python
    :caption: **References:** :meth:`~fortuna.data.loader.DataLoader.from_array_data`, :meth:`~fortuna.prob_model.classification.ProbClassifier.train`

    calib_data_loader = DataLoader.from_array_data(
        calib_data,
        batch_size=128,
        prefetch=True
    )
    status = prob_model.train(
        train_data_loader=train_data_loader,
        calib_data_loader=calib_data_loader
    )

Estimate statistics
-----------------------------
Given some test data :code:`test_data`,
which we will convert to a data loader like done above,
we are ready to estimate predictive statistics.
These include predictive mode, mean, log-pdf, variance, entropy, etc;
please consult the :ref:`predictive <predictive>` reference.
Apart from the log-pdf,
computing these statistics only require test input data,
never test target data.
With Fortuna,
you can easily construct a loader of test input data from a test data loader :code:`test_data_loader` by
typing :code:`test_data_loader.to_inputs_loader()`,
as you will see in the code below.

.. note::
    In classification, the predictive *mode* gives label predictions, i.e. the label
    predicted for a certain input, while the predictive *mean* gives probability predictions, i.e. the
    probability of each label.

.. code-block:: python
    :caption: **References:** :meth:`~fortuna.data.loader.DataLoader.from_array_data`, :meth:`~fortuna.data.loader.DataLoader.to_inputs_loader`, :meth:`~fortuna.prob_model.predictive.classification.ClassificationPredictive.log_prob`, :meth:`~fortuna.prob_model.predictive.classification.ClassificationPredictive.mode`, :meth:`~fortuna.prob_model.predictive.classification.ClassificationPredictive.mean`

    test_data_loader = DataLoader.from_array_data(
        test_data,
        batch_size=128
    )
    test_inputs_loader = test_data_loader.to_inputs_loader()
    test_logprob = prob_model.predictive.log_prob(
        data_loader=test_data_loader
    )
    test_modes = prob_model.predictive.mode(
        inputs_loader=test_inputs_loader
    )
    test_means = prob_model.predictive.mean(
        inputs_loader=test_inputs_loader
    )

Compute metrics
-----------------------------
Fortuna supports some classification metrics,
e.g. accuracy, expected calibration error and Brier score.
You are encouraged to bring in metrics from other frameworks and apply them on Fortuna's predictions,
as the latter are compatible with metrics operating on :code:`numpy.narray`.

Metrics often require arrays of test target data. You can easily get these by typing
:code:`test_data_loader.to_array_targets()`.

.. code-block:: python
    :caption: **References:** :meth:`~fortuna.data.loader.DataLoader.to_array_targets`, :func:`~fortuna.metric.classification.accuracy`, :func:`~fortuna.metric.classification.expected_calibration_error`

    from fortuna.metric.classification import accuracy, expected_calibration_error
    test_targets = test_data_loader.to_array_targets()
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
like in :ref:`conformal`,
starting from predictive statistics you can compute conformal sets.
Again, we need a data loader for this purpose.
For simplicity, we will use the same calibration data loader as above,
but a new one could be used.

.. code-block:: python
    :caption: **References:** :meth:`~fortuna.conformal.classification.adaptive_prediction.AdaptivePredictionConformalClassifier.conformal_set`

    from fortuna.conformal.classification import AdaptivePredictionConformalClassifier
    calib_inputs_loader = calib_data_loader.to_inputs_loader()
    calib_targets = calib_data_loader.to_array_targets()
    calib_means = prob_model.predictive.mean(
        inputs_loader=calib_inputs_loader
    )
    conformal_sets = AdaptivePredictionConformalClassifier().conformal_set(
        val_probs=calib_means,
        test_probs=test_means,
        val_targets=calib_targets
    )

.. _flax_models_regression:

Regression
================================
Similarly as in the :ref:`classification example <flax_models_classification>`,
let us show how to train a model, obtain prediction, compute metrics and conformal intervals in
regression.

Provide data loaders
-----------------------------
First, you are required to provide some training data :code:`train_data`. This must be
converted into a data loader compatible with Fortuna. In this example, we assume your data is a
tuple of arrays, respectively containing input and target variables. Several other data formats are
supported, e.g. PyTorch data loaders, TensorFlow data loader, etc. Please look into
:class:`~fortuna.data.loader.DataLoader` to
explore the supported conversion options.

.. code-block:: python
    :caption: **References:** :meth:`~fortuna.data.loader.DataLoader.from_array_data`

    from fortuna import DataLoader
    train_data_loader = DataLoader.from_array_data(
        train_data,
        batch_size=128,
        shuffle=True,
        prefetch=True
    )

Choose or build a model
-----------------------------
Second, you must provide a deep learning model mapping input variables to the space of the target variables.
You can either choose one of Fortuna's pre-built models, or build your own in
`Flax <https://flax.readthedocs.io/>`__. In this code example, we take a pre-built
Multi-Layer Perceptron (MLP), and denote the output dimension by :code:`output_dim`.

Additionally, you must build or choose a model for the log-variance of the likelihood function.
Let's build a linear one for this example.

.. code-block:: python
    :caption: **References:** :class:`~fortuna.prob_model.model.mlp.MLP`

    from fortuna.model import MLP
    model = MLP(
        output_dim=output_dim
    )
    likelihood_log_variance_model = MLP(
        output_dim=output_dim,
        widths=(),
        activations=()
    )

Build a probabilistic model
---------------------------
Fortuna now wraps the model and the likelihood log-variance model into a probabilistic regressor,
your interface object.
You may configure this by choosing a prior distribution,
a posterior approximation method,
an output calibration method and a random seed;
please consult the :ref:`probabilistic regressor <prob_regressor>` reference.
However, if you are not particularly familiar with these concepts, just go with the default options,
like we do in this example.

.. code-block:: python
    :caption: **References:** :class:`~fortuna.prob_model.regression.ProbRegressor`

    from fortuna import ProbRegressor
    prob_model = ProbRegressor(
        model=model,
        likelihood_log_variance_model=likelihood_log_variance_model
    )

Train the probabilistic model
-----------------------------
Let's train the probabilistic model now. Training includes two parts: fitting the posterior
distribution, and post-processing calibration. However, calibration is performed only if a calibration
data loader is provided. As for the training data, we assume you have calibration data
:code:`calib_data` formatted as a tuple of arrays, and let Fortuna transform it into a data loader. The
training returns a status object, describing the progress of the training process.

You are invited to configure fitting and calibration as you please. You can specify the optimization
process, saving and restoring checkpoints, monitoring the training and enable early stopping, and
select your computation device; please consult the :ref:`fit configuration <fit_config>` and
:ref:`calibration configuration <prob_model_calib_config>` references. In this example, we will stick with the
default configuration options.

.. code-block:: python
    :caption: **References:** :meth:`~fortuna.data.loader.DataLoader.from_array_data`, :meth:`~fortuna.prob_model.regression.ProbRegressor.train`

    calib_data_loader = DataLoader.from_array_data(calib_data, batch_size=128, prefetch=True)
    status = prob_model.train(
        train_data_loader=train_data_loader,
        calib_data_loader=calib_data_loader
    )

Estimate statistics
-----------------------------
Given some test data :code:`test_data`,
which we will convert to a data loader like done above,
we are ready to estimate predictive statistics.
These include predictive mode, mean, log-pdf, variance, entropy, quantile, credible interval, etc;
please consult the :ref:`predictive <predictive>` reference.
Apart from the log-pdf,
computing these statistics only require test input data,
never test target data.
With Fortuna,
you can easily construct a loader of input data from a test data loader :code:`test_data_loader` by
typing :code:`test_data_loader.to_inputs_loader()`,
as you will see in the code below.

.. note::
    In contrast with classification, in regression both the predictive *mean* and the predictive *mode* provide
    predictions for the target variables, and do not represent measures of uncertainty.

.. code-block:: python
    :caption: **References:** :meth:`~fortuna.data.loader.DataLoader.from_array_data`, :meth:`~fortuna.data.loader.DataLoader.to_inputs_loader`, :meth:`~fortuna.prob_model.predictive.regression.RegressionPredictive.log_prob`, :meth:`~fortuna.prob_model.predictive.regression.RegressionPredictive.mode`, :meth:`~fortuna.prob_model.predictive.regression.RegressionPredictive.mean`

    test_data_loader = DataLoader.from_array_data(
        test_data,
        batch_size=128
    )
    test_inputs_loader = test_data_loader.to_inputs_loader()
    test_logprob = prob_model.predictive.log_prob(
        data_loader=test_data_loader
    )
    test_means = prob_model.predictive.mean(
        inputs_loader=test_inputs_loader
    )
    test_cred_intervals = prob_model.predictive.credible_interval(
        inputs_loader=test_inputs_loader
    )

Compute metrics
-----------------------------
Fortuna supports some regression metrics,
e.g. Root Mean-Squared Error (RMSE) and Prediction Interval Coverage Probability (PICP).
You are encouraged to bring in metrics from other frameworks and apply them on Fortuna's predictions,
as the latter are compatible with metrics operating on :code:`numpy.ndarray`.

Metrics often require arrays of test target data. You can easily get these by typing
:code:`test_data_loader.to_array_targets()`.

.. code-block:: python
    :caption: **References:** :meth:`~fortuna.data.loader.DataLoader.to_array_targets`, :func:`~fortuna.metric.regression.root_mean_squared_error`, :func:`~fortuna.metric.regression.prediction_interval_coverage_probability`

    from fortuna.metric.regression import root_mean_squared_error, prediction_interval_coverage_probability
    test_targets = test_data_loader.to_array_targets()
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
Again, we need a data loader for this purpose.
For simplicity, we will use the same calibration data loader as above,
but a new one could be used.

.. code-block:: python
    :caption: **References:** :meth:`~fortuna.conformal.regression.quantile.QuantileConformalRegressor.conformal_interval`

    from fortuna.conformal.regression import QuantileConformalRegressor
    calib_inputs_loader = calib_data_loader.to_inputs_loader()
    calib_targets = calib_data_loader.to_array_targets()
    calib_cred_intervals = prob_model.predictive.credible_interval(
        inputs_loader=calib_inputs_loader
    )
    conformal_intervals = QuantileConformalRegressor().conformal_intervals(
        val_lower_bounds=calib_cred_intervals[:, 0],
        val_upper_bounds=calib_cred_intervals[:, 1],
        test_lower_bounds=test_cred_intervals[:, 0],
        test_upper_bounds=test_cred_intervals[:, 1],
        val_targets=calib_targets
    )

