Fortuna
#######

This project is archived and is no longer being actively developed or monitored.
################################################################################

.. image:: https://img.shields.io/pypi/status/Fortuna
    :target: https://img.shields.io/pypi/status/Fortuna
    :alt: PyPI - Status
.. image:: https://img.shields.io/pypi/dm/aws-fortuna
    :target: https://pypistats.org/packages/aws-fortuna
    :alt: PyPI - Downloads
.. image:: https://img.shields.io/pypi/v/aws-fortuna
    :target: https://img.shields.io/pypi/v/aws-fortuna
    :alt: PyPI - Version
.. image:: https://img.shields.io/github/license/awslabs/Fortuna
    :target: https://github.com/awslabs/Fortuna/blob/main/LICENSE
    :alt: License
.. image:: https://readthedocs.org/projects/aws-fortuna/badge/?version=latest
    :target: https://aws-fortuna.readthedocs.io
    :alt: Documentation Status

A Library for Uncertainty Quantification
========================================
Proper estimation of predictive uncertainty is fundamental in applications that involve critical decisions.
Uncertainty can be used to assess reliability of model predictions, trigger human intervention,
or decide whether a model can be safely deployed in the wild.

Fortuna is a library for uncertainty quantification that makes it easy for users to run benchmarks and bring uncertainty to production systems.
Fortuna provides calibration and conformal methods starting from pre-trained models written in any framework,
and it further supports several Bayesian inference methods starting from deep learning models written in `Flax <https://flax.readthedocs.io/en/latest/index.html>`_.
The language is designed to be intuitive for practitioners unfamiliar with uncertainty quantification,
and is highly configurable.

Check the `documentation <https://aws-fortuna.readthedocs.io/en/latest/>`_ for a quickstart, examples and references.

Usage modes
===========
Fortuna offers three different usage modes:
`From uncertainty estimates <https://github.com/awslabs/fortuna#from-uncertainty-estimates>`_,
`From model outputs <https://github.com/awslabs/fortuna#from-model-outputs>`_ and
`From Flax models <https://github.com/awslabs/fortuna#from-flax-models>`_.
These serve users according to the constraints dictated by their own applications.
Their pipelines are depicted in the following figure, each starting from one of the green panels.

.. image:: https://github.com/awslabs/fortuna/raw/main/docs/source/_static/pipeline.png
    :target: https://github.com/awslabs/fortuna/raw/main/docs/source/_static/pipeline.png

From uncertainty estimates
---------------------------
Starting from uncertainty estimates has minimal compatibility requirements and it is the quickest level of interaction with the library.
This usage mode offers conformal prediction methods for both classification and regression.
These take uncertainty estimates in input,
and return rigorous sets of predictions that retain a user-given level of probability.
In one-dimensional regression tasks, conformal sets may be thought as calibrated versions of confidence or credible intervals.

Mind that if the uncertainty estimates that you provide in inputs are inaccurate,
conformal sets might be large and unusable.
For this reason, if your application allows it,
please consider the `From model outputs <https://github.com/awslabs/fortuna#from-model-outputs>`_ and
`From Flax models <https://github.com/awslabs/fortuna#from-flax-models>`_ usage modes.

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

 from fortuna.conformal import QuantileConformalRegressor
 conformal_intervals = QuantileConformalRegressor().conformal_interval(
      val_lower_bounds=val_lower_bounds, val_upper_bounds=val_upper_bounds,
      test_lower_bounds=test_lower_bounds, test_upper_bounds=test_upper_bounds,
      val_targets=val_targets, error=error)

From model outputs
------------------
Starting from model outputs assumes you have already trained a model in some framework,
and arrive to Fortuna with model outputs in :code:`numpy.ndarray` format for each input data point.
This usage mode allows you to calibrate your model outputs, estimate uncertainty,
compute metrics and obtain conformal sets.

Compared to the `From uncertainty estimates <https://github.com/awslabs/fortuna#from-uncertainty-estimates>`_ usage mode,
this one offers better control,
as it can make sure uncertainty estimates have been appropriately calibrated.
However, if the model had been trained with classical methods,
the resulting quantification of model (a.k.a. epistemic) uncertainty may be poor.
To mitigate this problem, please consider the `From Flax models <https://github.com/awslabs/fortuna#from-flax-models>`_
usage mode.

**Example.**
Suppose you have validation and test model outputs,
respectively :code:`val_outputs` and :code:`test_outputs`.
Furthermore, you have some arrays of validation and target variables,
respectively :code:`val_targets` and :code:`test_targets`.
The following code provides a minimal classification example to get calibrated predictive entropy estimates.

.. code-block:: python

  from fortuna.output_calib_model import OutputCalibClassifier
  calib_model = OutputCalibClassifier()
  status = calib_model.calibrate(outputs=val_outputs, targets=val_targets)
  test_entropies = calib_model.predictive.entropy(outputs=test_outputs)

From Flax models
--------------------------
Starting from Flax models has higher compatibility requirements than the
`From uncertainty estimates <https://github.com/awslabs/fortuna#from-uncertainty-estimates>`_
and `From model outputs <https://github.com/awslabs/fortuna#from-model-outputs>`_ usage modes,
as it requires deep learning models written in `Flax <https://flax.readthedocs.io/en/latest/index.html>`_.
However, it enables you to replace standard model training with scalable Bayesian inference procedures,
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

Alternatively, you can build the package using `Poetry <https://python-poetry.org/docs/>`_.
If you choose to pursue this way, first install Poetry and add it to your PATH
(see `here <https://python-poetry.org/docs/#installation>`_). Then type

.. code-block::

    poetry install

All the dependencies will be installed at their required versions. Consider adding the following flags to the command above:

- :code:`-E transformers` if you want to use models and datasets from `Hugging Face <https://huggingface.co/>`_.
- :code:`-E sagemaker` if you want to install the dependencies necessary to run Fortuna on Amazon SageMaker.
- :code:`-E docs` if you want to install Sphinx dependencies to build the documentation.
- :code:`-E notebooks` if you want to work with Jupyter notebooks.

Finally, you can either access the virtualenv that Poetry created by typing :code:`poetry shell`,
or execute commands within the virtualenv using the :code:`run` command, e.g. :code:`poetry run python`.

Examples
========
Several usage examples are found in the
`/examples <https://github.com/awslabs/fortuna/tree/main/examples>`_
directory.

Amazon SageMaker integration
============================
We offer a simple pipeline that allows you to run Fortuna on Amazon SageMaker with minimal effort.

1. Create an AWS account - it is free! Store the account ID and the region where you want to launch training jobs.

2. First, `update your local AWS credentials <https://docs.aws.amazon.com/cli/latest/userguide/cli-authentication-short-term.html>`_.
   Then you need to build and `push a Docker image to an Amazon ECR repository <https://docs.aws.amazon.com/AmazonECR/latest/userguide/docker-push-ecr-image.html>`_.
   This `script <https://github.com/awslabs/fortuna/tree/main/fortuna/docker/build_and_push.sh>`_ will help you doing so -
   it will require your AWS account ID and region. If you need other packages to be included in your Docker image,
   you should consider customize the `Dockerfile <https://github.com/awslabs/fortuna/tree/main/fortuna/docker/Dockerfile>`_.
   NOTE: the script has been tested on a M1 MacOS.
   It is possible that different operating systems will need small modifications.

3. Create an `S3 bucket <https://docs.aws.amazon.com/AmazonS3/latest/userguide/creating-bucket.html>`_.
   You will need this to dump the results from your training jobs on Amazon Sagemaker.

4. Write a configuration `yaml` file. This will include your AWS details, the path to the entrypoint script that you want
   to run on Amazon SageMaker, the arguments to pass to the script, the path to the S3 bucket where you want to dump
   the results, the metrics to monitor, and more.
   Check `this file <https://github.com/awslabs/fortuna/tree/main/benchmarks/transformers/sagemaker_entrypoints/prob_model_text_classification_config/default.yaml>`_ for an example.

5. Finally, given :code:`config_dir`, that is the absolute path to the main configuration directory,
   and :code:`config_filename`, that is the name of the main configuration file (without .yaml extension),
   enter Python and run the following:

.. code-block:: python

    from fortuna.sagemaker import run_training_job
    run_training_job(config_dir=config_dir, config_filename=config_filename)


Material
========
- `AWS launch blog post <https://aws.amazon.com/blogs/machine-learning/introducing-fortuna-a-library-for-uncertainty-quantification/>`_
- `Fortuna: A Library for Uncertainty Quantification in Deep Learning [arXiv paper] <https://arxiv.org/abs/2302.04019>`_

Citing Fortuna
==============
To cite Fortuna:

.. code-block::

    @article{detommaso2023fortuna,
      title={Fortuna: A Library for Uncertainty Quantification in Deep Learning},
      author={Detommaso, Gianluca and Gasparin, Alberto and Donini, Michele and Seeger, Matthias and Wilson, Andrew Gordon and Archambeau, Cedric},
      journal={arXiv preprint arXiv:2302.04019},
      year={2023}
    }

Contributing
============
If you wish to contribute to the project, please refer to our `contribution guidelines <https://github.com/awslabs/fortuna/blob/main/CONTRIBUTING.md>`_.


License
=======
This project is licensed under the Apache-2.0 License.
See `LICENSE <https://github.com/awslabs/fortuna/blob/main/LICENSE>`_ for more information.
