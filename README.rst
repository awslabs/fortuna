Fortuna
#######
A Library for Uncertainty Quantification
========================================
Proper estimation of predictive uncertainty is fundamental in applications that involve critical decisions.
It can be used to assess reliability of model predictions, trigger human intervention,
or decide whether a model can be safely deployed in the wild.

Fortuna provides calibrated uncertainty estimates of model predictions, in classification and regression.
It is designed to be easy-to-use,
and to promote effortless estimation of uncertainty in production systems.

Check the [documentation] for quickstart instructions, examples and API references.

Usage modes
===========
Fortuna offers several usage modes.
Their pipeline is depicted in the figure below.
You can start your journey with Fortuna from any of the green panels.

.. figure:: docs/source/pipeline.png


You might prefer one usage mode or the other according to your own requirements.

- Starting **from uncertainty estimates** has minimal compatibility requirements and it is the quickest level of interaction with the library.
  However, if the uncertainty estimates you provide are inaccurate,
  conformal sets (i.e. rigorous sets of likely predictions) might be large and unusable.

   **Example.** Suppose you want to calibrate credible intervals with coverage error :code:`error`,
   each corresponding to a different test input variable.
   We assume that credible intervals are passed as arrays of lower
   and upper bounds,
   respectively :code:`test_lower_bounds` and :code:`test_upper_bounds`.
   You also have lower and upper bounds of credible intervals computed for several validation inputs,
   respectively :code:`val_lower_bounds` and :code:`val_upper_bounds`.
   The corresponding array of validation targets is denoted by :code:`val_targets`.
   The following code produces *conformal prediction intervals*,
   i.e. calibrated versions of you test credible intervals.

   .. code-block:: python

     from fortuna.conformer.regression import QuantileConformalRegressor
     conformal_intervals = QuantileConformalRegressor().conformal_interval(
          val_lower_bounds=val_lower_bounds, val_upper_bounds=val_upper_bounds,
          test_lower_bounds=test_lower_bounds, test_upper_bounds=test_upper_bounds,
          val_targets=val_targets, error=error)

- Starting **from model outputs** offers better control,
  as it can make sure uncertainty estimates have been appropriately calibrated.
  However, these may have been obtain with classical training methods,
  that may not capture model (epistemic) uncertainty sufficiently well.

   **Example.**
   Suppose you have validation and test model outputs,
   respectively :code:`val_outputs` and :code:`test_outputs`.
   Furthermore, you have some arrays of validation and target variables,
   respectively :code:`val_targets` and :code:`test_targets`.
   The following code provides a minimal classification example to get calibrated predictive entropy estimates.

   .. code-block:: python

      from fortuna.calib_model import CalibClassifier
      calib_model = CalibClassifier()
      status = calib_model.calibrate(outputs=val_outputs, targets=val_targets)
      test_entropies = calib_model.predictive.entropy(outputs=test_outputs)

- Starting **from Flax models** has higher compatibility requirements,
  as it requires you to build a deep learning model in `Flax <https://flax.readthedocs.io/en/latest/index.html>`_.
  However, it enables you to replace standard training with scalable Bayesian inference procedures,
  which may significantly improve the quantification of predictive uncertainty.

   **Example.** Suppose you have a Flax classification deep learning model :code:`model` from inputs to logits, with output
   dimension given by :code:`output_dim`. Furthermore,
   you have some training, validation and calibration TensorFlow data loader :code:`train_data_loader`, :code:`val_data_loader`
   and :code:`test_data_loader`, respectively.
   The following code provides a minimal classification example to get calibrated probability estimates.

   .. code-block:: python

      from fortuna.data import DataLoader
      train_data_loader = DataLoader.from_tensorflow_data_loader(train_data_loader)
      calib_data_loader = DataLoader.from_tensorflow_data_loader(val_data_loader)
      test_data_loader = DataLoader.from_tensorflow_data_loader(test_data_loader)

      from fortuna.prob_model import ProbClassifier
      prob_model = ProbClassifier(model=model)
      status = prob_model.train(train_data_loader=train_data_loader, calib_data_loader=calib_data_loader)
      test_means = prob_model.predictive.mean(inputs_loader=test_data_loader.to_inputs_loader())


Installation
============
**NOTE:** Before installing Fortuna, you are required to `install JAX <https://github.com/google/jax#installation>`_ in your virtual environment.

You can install Fortuna by typing

.. code-block::

    pip install aws-fortuna

License
=======
This project is licensed under the Apache-2.0 License.
See `LICENSE <https://github.com/awslabs/fortuna/blob/main/LICENSE>`_ for more information.

Security
========
See `CONTRIBUTING.md <https://github.com/awslabs/fortuna/blob/main/CONTRIBUTING.md>`_ for more information.