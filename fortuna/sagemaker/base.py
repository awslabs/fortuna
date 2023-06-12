import logging
import pathlib

import boto3
from hydra import (
    compose,
    initialize_config_dir,
)
from hydra.utils import instantiate
import sagemaker
from sagemaker.debugger import (
    FrameworkProfile,
    ProfilerConfig,
    ProfilerRule,
    rule_configs,
)
from sagemaker.tensorflow import TensorFlow
from sagemaker.tuner import HyperparameterTuner

import fortuna
from fortuna.sagemaker.utils import (
    DictConfig,
    create_input_channels,
    get_base_job_name,
    get_hparams,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _run_training_job(cfg: DictConfig) -> None:
    """
    Run a training job on Amazon SageMaker.
    The model, the hyperparameters, and all the sagemaker-related parameters
    (image_uri to be used, the model training entry point, the AWS IAM role to use, the instance type/count to use,
    the input/output paths on s3 or local disk) have to be defined in the configuration dictionary.

    Parameters
    ----------
    cfg: DictConfig
        Configuration dictionary.
    """
    if hasattr(cfg.sagemaker, "metrics"):
        metrics = [dict(d) for d in cfg.sagemaker.metrics]
    else:
        metrics = []

    # define the dependency path
    source_dir = pathlib.Path(fortuna.__file__).parents[1]
    # get proper datasets
    channels = create_input_channels(cfg)
    # get entrypoint args
    hparams = get_hparams(cfg=cfg)
    # update channels
    if "model_path" in cfg.model:
        channels.update({"loadmodel": cfg.model.model_path})

    logger.info(f"Training job with parameters: {hparams}")
    logger.info(f"\nEstimating with channels: {channels}")

    base_job_name = get_base_job_name(cfg)

    logger.info(f"Source dir: {source_dir}")

    rules = [
        ProfilerRule.sagemaker(rule_configs.LowGPUUtilization()),
        ProfilerRule.sagemaker(rule_configs.ProfilerReport()),
    ]
    profiler_config = ProfilerConfig(
        system_monitor_interval_millis=500,
        framework_profile_params=FrameworkProfile(
            local_path="/opt/ml/output/profiler/", start_step=5, num_steps=2
        ),
    )

    sm_role = f"arn:aws:iam::{cfg.sagemaker.account_id}:role/{cfg.sagemaker.iam_role}"
    boto_sess = boto3.Session(
        profile_name=cfg.sagemaker.profile, region_name=cfg.sagemaker.region
    )
    sm_sess = sagemaker.Session(boto_session=boto_sess)
    sm = boto_sess.client(service_name="sagemaker")

    estimator = TensorFlow(
        image_uri=f"{cfg.sagemaker.account_id}.dkr.ecr.{cfg.sagemaker.region}.amazonaws.com/fortuna:dev",
        entry_point=f"./{cfg.sagemaker.entrypoint}",
        source_dir=str(source_dir),
        role=sm_role,
        instance_type=cfg.sagemaker.instance_type,
        instance_count=1,
        hyperparameters=hparams,
        output_path=cfg.output_data_path,
        base_job_name=base_job_name,
        metric_definitions=metrics,
        py_version="py38",
        sagemaker_session=sm_sess,
        profiler_config=profiler_config,
        rules=rules,
    )

    if "tuner" in cfg.sagemaker:
        logger.info(f"Starting hyperparams optimization: {cfg.sagemaker.tuner}")
        estimator = HyperparameterTuner(
            estimator=estimator,
            base_tuning_job_name=base_job_name,
            metric_definitions=metrics,
            **instantiate(cfg.sagemaker.tuner, _convert_="partial"),
        )

    estimator.fit(inputs=channels, wait=False)


def run_training_job(config_dir: str, config_filename: str):
    """
    Run a training job on Amazon SageMaker starting from a path to the configuration directory, and the name of the main configuration file.
    All configuration files should be written in `yaml` format. The following information must be provided.

    .. code-block:: yaml

        sagemaker:
          account_id: ~
          iam_role: ~
          entrypoint: ~
          instance_type: ~
          profile: ~
          region: ~

        dataset:
          base_data_path: ~

        task:
          name: ~

        model:
          name: ~

        method:
          name: ~

    `sagemaker` may optionally include `metrics`, which specifies the `training metrics <https://docs.aws.amazon.com/sagemaker/latest/dg/training-metrics.html>`_ to be tracked by SageMaker,
    and `tuner`, which configures the `hyperparameter tuning <https://sagemaker.readthedocs.io/en/stable/api/training/tuner.html>`_.
    Any argument to pass to the entrypoint script must be specified under `hparams` in either `task`, `model` or `method`.
    For example:

    .. code-block:: yaml

        model:
          name: ~
            hparams:
              some_hyperparameter: ~

    For the training job to run successfully, you must have already built and pushed a Docker imagine to AWS in the
    region specified by the configuration file. You can do so by running `bash fortuna/docker/build_and_push.sh`,
    providing the required information as argument. Docker and related plug-ins may need to be installed beforehand.

    Parameters
    ----------
    config_dir: str
        Absolute path to main configuration directory.
    config_filename: str
        Main configuration filename. This should not include the `.yaml` extension.
    """
    with initialize_config_dir(config_dir=config_dir):
        cfg = compose(config_name=config_filename)
    _run_training_job(cfg)
