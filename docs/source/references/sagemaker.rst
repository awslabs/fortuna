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

3. Write a configuration `yaml` file. This will include your AWS details, the path to the entrypoint script that you want
   to run on Amazon SageMaker, the arguments to pass to the script, the path to the S3 bucket where you want to dump
   the results, the metrics to monitor, and more.
   See `here <https://github.com/awslabs/fortuna/tree/main/benchmarks/transformers/sagemaker_entrypoints/prob_model_text_classification_config/default.yaml>`_ for an example.

4. Finally, given :code:`config_dir`, that is the absolute path to the main configuration directory,
   and :code:`config_filename`, that is the name of the main configuration file (without .yaml extension),
   enter Python and run the following:

.. code-block:: python

    from fortuna.sagemaker import run_training_job
    run_training_job(config_dir=config_dir, config_filename=config_filename)


.. autofunction:: fortuna.sagemaker.base.run_training_job
