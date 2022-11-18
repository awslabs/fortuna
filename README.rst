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

- Starting **from model outputs** offers better control,
  as it can make sure uncertainty estimates have been appropriately calibrated.
  However, these may have been obtain with classical training methods,
  that may not capture model (epistemic) uncertainty sufficiently well.

- Starting **from Flax models** has higher compatibility requirements,
  as it requires you to build a deep learning model in `Flax <https://flax.readthedocs.io/en/latest/index.html>`_.
  However, it enables you to replace standard training with scalable Bayesian inference procedures,
  which may significantly improve the quantification of predictive uncertainty.

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