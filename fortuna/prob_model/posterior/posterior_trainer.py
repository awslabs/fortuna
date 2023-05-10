from fortuna.prob_model.posterior.posterior_mixin import WithPosteriorCheckpointingMixin
from fortuna.training.trainer import TrainerABC


class PosteriorTrainerABC(WithPosteriorCheckpointingMixin, TrainerABC):
    pass
