import logging
from typing import (
    Optional,
    Tuple,
)

from fortuna.data.loader import DataLoader
from fortuna.prob_model.fit_config.base import FitConfig
from fortuna.prob_model.joint.base import Joint
from fortuna.prob_model.posterior.map.map_approximator import MAPPosteriorApproximator
from fortuna.prob_model.posterior.map.map_posterior import MAPPosterior
from fortuna.prob_model.posterior.map.map_state import MAPState
from fortuna.typing import Status
from fortuna.utils.random import RandomNumberGenerator
from fortuna.partitioner.partition_manager.base import PartitionManager


def run_preliminary_map(
    joint: Joint,
    partition_manager: PartitionManager,
    train_data_loader: DataLoader,
    val_data_loader: DataLoader,
    map_fit_config: Optional[FitConfig],
    rng: RandomNumberGenerator,
    **kwargs,
) -> Tuple[MAPState, Status]:
    logging.info("Do a preliminary run of MAP.")
    map_posterior = MAPPosterior(
        joint, posterior_approximator=MAPPosteriorApproximator(), partition_manager=partition_manager
    )
    map_posterior.rng = rng
    status = map_posterior.fit(
        rng=rng.get(),
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        fit_config=map_fit_config,
        **kwargs,
    )
    logging.info("Preliminary run with MAP completed.")
    return map_posterior.state.get(), status
