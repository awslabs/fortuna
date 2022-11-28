import logging
from typing import Type

import jax


def select_trainer_given_devices(
    devices: int,
    BaseTrainer: Type,
    JittedTrainer: Type,
    MultiDeviceTrainer: Type,
    disable_jit: bool,
) -> Type:
    if devices not in [0, -1]:
        raise NotImplementedError(
            "Currently, only two options are supported: use all available (`devices=-1`) or use only CPU (`devices=0`)."
        )
    elif devices == -1 and disable_jit:
        logging.warning("Jit must be enabled when not training on a single CPU device.")

    if devices == -1:
        logging.info("Training on all available devices.")
        trainer_cls = (
            MultiDeviceTrainer
            if len([d for d in jax.devices() if d.platform == "device"]) > 0
            else JittedTrainer
        )

    elif devices == 0 and disable_jit:
        logging.info("Training on CPU without jit.")
        trainer_cls = BaseTrainer
    else:
        logging.info("Training on CPU.")
        trainer_cls = JittedTrainer
    return trainer_cls
