import logging
from fortuna.prob_model.posterior.map.map_posterior import MAPPosterior
from fortuna.prob_model.posterior.map.map_approximator import MAPPosteriorApproximator
from fortuna.typing import Status
from typing import Tuple, Optional
from fortuna.prob_model.joint.base import Joint
from fortuna.data.loader import DataLoader
from fortuna.utils.random import RandomNumberGenerator
from fortuna.prob_model.fit_config.base import FitConfig
from fortuna.prob_model.posterior.map.map_state import MAPState


def run_preliminary_map(
        joint: Joint,
        train_data_loader: DataLoader,
        val_data_loader: DataLoader,
        map_fit_config: Optional[FitConfig],
        rng: RandomNumberGenerator,
        **kwargs
) -> Tuple[MAPState, Status]:
    logging.info("Do a preliminary run of MAP.")
    map_posterior = MAPPosterior(
        joint, posterior_approximator=MAPPosteriorApproximator()
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
