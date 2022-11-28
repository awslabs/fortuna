Quickstart
===========
Fortuna offers three different usage modes:
:ref:`from_uncertainty_estimates`,
:ref:`from_model_outputs` and
:ref:`from_flax_models`.
These serve users according to the constraints dictated by their own applications.
Their pipelines are depicted in the following figure, each starting from one of the green panels.

.. figure:: _static/pipeline.png

The following sections offer a glance over each of the usage modes.
See :ref:`usage_modes` for more details.

.. _from_uncertainty_estimates:

From uncertainty estimates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Starting from uncertainty estimates has minimal compatibility requirements and it is the quickest level of interaction with the library.
This usage mode offers conformal prediction methods for both classification and regression.
These take uncertainty estimates in input,
and return rigorous sets of predictions that retain a user-given level of probability.
In one-dimensional regression tasks, conformal sets may be thought as calibrated intervals of confidence or credible intervals.

Mind that if the uncertainty estimates that you provide in inputs are inaccurate,
conformal sets might be large and unusable.
For this reason, if your application allows it,
please consider the :ref:`from_model_outputs` and
:ref:`from_flax_models` usage modes.

**Example.** Suppose you want to calibrate credible intervals with coverage error :code:`error`,
each corresponding to a different test input variable.
We assume that credible intervals are passed as arrays of lower and upper bounds,
respectively :code:`test_lower_bounds` and :code:`test_upper_bounds`.
You also have lower and upper bounds of credible intervals computed for several validation inputs,
respectively :code:`val_lower_bounds` and :code:`val_upper_bounds`.
The corresponding array of validation targets is denoted by :code:`val_targets`.
The following code produces *conformal prediction intervals*,
i.e. calibrated versions of you test credible intervals.

.. code-block:: python
      :caption: **References:** :meth:`~fortuna.conformal.regression.QuantileConformalRegressor.conformal_interval`

      from fortuna.conformal.regression import QuantileConformalRegressor
      conformal_intervals = QuantileConformalRegressor().conformal_interval(
            val_lower_bounds=val_lower_bounds, val_upper_bounds=val_upper_bounds,
            test_lower_bounds=test_lower_bounds, test_upper_bounds=test_upper_bounds,
            val_targets=val_targets, error=error)

.. _from_model_outputs:

From model outputs
~~~~~~~~~~~~~~~~~~
Starting from model outputs assumes you have already trained a model in some framework,
and arrive to Fortuna with model outputs in :code:`numpy.ndarray` format for each input data point.
This usage mode allows you to calibrate your model outputs, estimate uncertainty,
compute metrics and obtain conformal sets.

Compared to the :ref:`from_uncertainty_estimates` usage mode,
this one offers better control,
as it can make sure uncertainty estimates have been appropriately calibrated.
However, if the model had been trained with classical methods,
the resulting quantification of model (a.k.a. epistemic) uncertainty may be poor.
To mitigate this problem, please consider the :ref:`from_flax_models`
usage mode.

**Example.**
Suppose you have validation and test model outputs,
respectively :code:`val_outputs` and :code:`test_outputs`.
Furthermore, you have some arrays of validation and target variables,
respectively :code:`val_targets` and :code:`test_targets`.
The following code provides a minimal classification example to get calibrated predictive entropy estimates.

.. code-block:: python
      :caption: **References:** :class:`~fortuna.calib_model.classification.CalibClassifier`, :meth:`~fortuna.calib_model.classification.CalibClassifier.calibrate`, :meth:`~fortuna.calib_model.predictive.classification.ClassificationPredictive.entropy`

      from fortuna.calib_model import CalibClassifier
      calib_model = CalibClassifier()
      status = calib_model.calibrate(outputs=val_outputs, targets=val_targets)
      test_entropies = calib_model.predictive.entropy(outputs=test_outputs)

.. _from_flax_models:

From Flax models
~~~~~~~~~~~~~~~~
Starting from Flax models has higher compatibility requirements than the
:ref:`from_uncertainty_estimates` and :ref:`from_model_outputs` usage modes,
as it requires deep learning models written in `Flax <https://flax.readthedocs.io/en/latest/index.html>`_.
However, it enables you to replace standard model training with scalable Bayesian inference procedures,
which may significantly improve the quantification of predictive uncertainty.

**Example.** Suppose you have a Flax classification deep learning model :code:`model` from inputs to logits, with output
dimension given by :code:`output_dim`. Furthermore,
you have some training, validation and calibration TensorFlow data loader :code:`train_data_loader`, :code:`val_data_loader`
and :code:`test_data_loader`, respectively.
The following code provides a minimal classification example to get calibrated probability estimates.

.. code-block:: python
      :caption: **References:** :meth:`~fortuna.data.loader.DataLoader.from_tensorflow_data_loader`, :class:`~fortuna.prob_model.classification.ProbClassifier`, :meth:`~fortuna.prob_model.classification.ProbClassifier.train`, :meth:`~fortuna.prob_model.predictive.classification.ClassificationPredictive.mean`

      from fortuna.data import DataLoader
      train_data_loader = DataLoader.from_tensorflow_data_loader(train_data_loader)
      calib_data_loader = DataLoader.from_tensorflow_data_loader(val_data_loader)
      test_data_loader = DataLoader.from_tensorflow_data_loader(test_data_loader)

      from fortuna.prob_model import ProbClassifier
      prob_model = ProbClassifier(model=model)
      status = prob_model.train(train_data_loader=train_data_loader, calib_data_loader=calib_data_loader)
      test_means = prob_model.predictive.mean(inputs_loader=test_data_loader.to_inputs_loader())
