from fortuna.prob_model.posterior.normalizing_flow.advi import ADVI_NAME
from fortuna.prob_model.posterior.normalizing_flow.normalizing_flow_trainer import \
    NormalizingFlowTrainer


class ADVITrainer(NormalizingFlowTrainer):
    def __str__(self):
        return ADVI_NAME
