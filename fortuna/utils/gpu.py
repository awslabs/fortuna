import logging
from typing import Type

import jax


def select_trainer_given_devices(
    gpus: int,
    BaseTrainer: Type,
    JittedTrainer: Type,
    MultiGPUTrainer: Type,
    disable_jit: bool,
) -> Type:
    if gpus not in [0, -1]:
        raise NotImplementedError(
            "Currently, only two options are supported: use all available (`gpus=-1`) or use no GPU (`gpus=0`)."
        )
    elif gpus == -1 and disable_jit:
        logging.warning("When training on GPUs, jit cannot be disabled.")

    if gpus == -1:
        logging.info("Training on all the available GPUs, if any.")
        trainer_cls = (
            MultiGPUTrainer
            if len([d for d in jax.devices() if d.platform == "gpu"]) > 0
            else JittedTrainer
        )

    elif gpus == 0 and disable_jit:
        logging.info("Training on CPU without jit.")
        trainer_cls = BaseTrainer
    else:
        logging.info("Training on CPU.")
        trainer_cls = JittedTrainer
    return trainer_cls
