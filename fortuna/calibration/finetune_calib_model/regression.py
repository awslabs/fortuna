from fortuna.calibration.finetune_calib_model.base import FinetuneCalibModel
from fortuna.calibration.finetune_calib_model.predictive.regression import RegressionPredictive
from fortuna.prob_output_layer.regression import RegressionProbOutputLayer
from fortuna.model.model_manager.regression import RegressionModelManager
from fortuna.likelihood.regression import RegressionLikelihood
from fortuna.typing import Path, Status
from flax import linen as nn
from fortuna.data import DataLoader
from fortuna.calibration.finetune_calib_model.config.base import Config
from fortuna.calibration.loss.base import Loss
from typing import Optional


class FinetuneCalibRegressor(FinetuneCalibModel):
    def __init__(
            self,
            model: nn.Module,
            likelihood_log_variance_model: nn.Module,
            restore_checkpoint_path: Path,
            seed: int = 0):
        self.model_manager = RegressionModelManager(model, likelihood_log_variance_model)
        self.prob_output_layer = RegressionProbOutputLayer()
        self.predictive = RegressionPredictive(
            likelihood=RegressionLikelihood(
                model_manager=self.model_manager,
                prob_output_layer=self.prob_output_layer,
                output_calib_manager=None
            ),
            restore_checkpoint_path=restore_checkpoint_path
        )
        super().__init__(seed=seed)

    def _check_output_dim(self, data_loader: DataLoader):
        data_output_dim = 0
        for x, y in data_loader:
            input_shape = x.shape[1:]
            data_output_dim = y.shape[1]
            break
        if data_output_dim == 0:
            raise ValueError(
                """`_data_loader` is either empty or incorrectly constructed."""
            )
        model_manager_output_dim = self._get_output_dim(input_shape)
        if model_manager_output_dim != 2 * data_output_dim:
            raise ValueError(
                f"""The outputs dimension of both `model` and `likelihood_log_variance_model` must be the same as
                the dimension of the target variables in `_data_loader`. However, {model_manager_output_dim // 2} and 
                {data_output_dim} were found, respectively."""
            )

    def calibrate(
        self,
        calib_data_loader: DataLoader,
        val_data_loader: Optional[DataLoader] = None,
        config: Config = Config(),
        loss_fn: Optional[Loss] = None
    ) -> Status:
        self._check_output_dim(calib_data_loader)
        if val_data_loader is not None:
            self._check_output_dim(val_data_loader)
        return self._calibrate(
            calib_data_loader=calib_data_loader,
            uncertainty_fn=config.monitor.uncertainty_fn
            if config.monitor.uncertainty_fn is not None
            else self.prob_output_layer.variance,
            val_data_loader=val_data_loader,
            config=config,
            loss_fn=loss_fn
        )

