from fortuna.calibration.calib_model.base import CalibModel
from fortuna.calibration.calib_model.predictive.regression import RegressionPredictive
from fortuna.prob_output_layer.regression import RegressionProbOutputLayer
from fortuna.model.model_manager.regression import RegressionModelManager
from fortuna.likelihood.regression import RegressionLikelihood
from fortuna.typing import Status, Outputs, Targets
from fortuna.loss.regression.scaled_mse import scaled_mse_fn
from flax import linen as nn
from fortuna.data import DataLoader
from fortuna.calibration.calib_model.config.base import Config
from typing import Optional, Callable
import jax.numpy as jnp


class CalibRegressor(CalibModel):
    def __init__(
            self,
            model: nn.Module,
            likelihood_log_variance_model: nn.Module,
            seed: int = 0):
        self.model_manager = RegressionModelManager(model, likelihood_log_variance_model)
        self.prob_output_layer = RegressionProbOutputLayer()
        self.likelihood = RegressionLikelihood(
            model_manager=self.model_manager,
            prob_output_layer=self.prob_output_layer,
            output_calib_manager=None
        )
        self.predictive = RegressionPredictive(
            likelihood=self.likelihood
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
        loss_fn: Callable[[Outputs, Targets], jnp.ndarray] = scaled_mse_fn,
        config: Config = Config(),
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
            loss_fn=loss_fn,
            config=config,
        )

