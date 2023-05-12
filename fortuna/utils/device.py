import logging
from typing import Type

import jax


def select_trainer_given_devices(
    devices: int,
    base_trainer_cls: Type,
    jitted_trainer_cls: Type,
    multi_device_trainer_cls: Type,
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
            multi_device_trainer_cls
            if len([d for d in jax.devices() if d.platform == "gpu"]) > 0
            else jitted_trainer_cls
        )

    elif devices == 0 and disable_jit:
        logging.info("Training on CPU without jit.")
        trainer_cls = base_trainer_cls
    else:
        logging.info("Training on CPU.")
        trainer_cls = jitted_trainer_cls
    return trainer_cls
