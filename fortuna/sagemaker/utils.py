from typing import (
    Any,
    Dict,
)

from omegaconf import DictConfig


def create_input_channels(cfg: DictConfig) -> Dict[str, str]:
    base_path = cfg.dataset.base_data_path
    base_path = base_path if base_path.endswith("/") else base_path + "/"

    if (
        hasattr(cfg.dataset, "train_relative_path")
        and cfg.dataset.train_relative_path != ""
    ):
        channels = {"train": base_path + cfg.dataset.train_relative_path}
    else:
        channels = {"train": base_path}
    if (
        hasattr(cfg.dataset, "test_relative_path")
        and cfg.dataset.test_relative_path != ""
    ):
        channels.update({"test": base_path + cfg.dataset.test_relative_path})
    if (
        hasattr(cfg.dataset, "validation_relative_path")
        and cfg.dataset.validation_relative_path != ""
    ):
        channels.update(
            {"validation": base_path + cfg.dataset.validation_relative_path}
        )
    return channels


def get_base_job_name(cfg: DictConfig) -> str:
    base_job_name = (
        f"fortuna-{cfg.task.name}-{cfg.method.name}-{cfg.model.name}".replace("_", "-")
    )
    if cfg.sagemaker.job_name_suffix is not None:
        base_job_name += f"-{cfg.sagemaker.job_name_suffix}".replace("_", "-")
    return base_job_name


def get_hparams(cfg: DictConfig) -> Dict[str, Any]:
    task_hparams = {k: v for k, v in cfg.task.hparams.items()}
    model_hparams = {k: v for k, v in cfg.model.hparams.items()}
    method_hparams = {k: v for k, v in cfg.method.hparams.items()}

    hparams = dict(**task_hparams, **model_hparams, **method_hparams)

    return hparams
