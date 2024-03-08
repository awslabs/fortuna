from typing import (
    Callable,
    Optional,
)

from flax import linen as nn
import jax.numpy as jnp

from fortuna.calib_model.base import CalibModel
from fortuna.calib_model.config.base import Config
from fortuna.calib_model.predictive.regression import RegressionPredictive
from fortuna.data import DataLoader
from fortuna.likelihood.regression import RegressionLikelihood
from fortuna.loss.regression.scaled_mse import scaled_mse_fn
from fortuna.model.model_manager.regression import RegressionModelManager
from fortuna.model_editor.base import ModelEditor
from fortuna.prob_output_layer.regression import RegressionProbOutputLayer
from fortuna.typing import (
    Outputs,
    Status,
    Targets,
)


class CalibRegressor(CalibModel):
    def __init__(
        self,
        model: nn.Module,
        likelihood_log_variance_model: nn.Module,
        model_editor: Optional[ModelEditor] = None,
        seed: int = 0,
    ):
        r"""
        A calibration regressor class.

        Parameters
        ----------
        model : nn.Module
            A model describing the deterministic relation between inputs and outputs. It characterizes the mean model
            of the likelihood function. The outputs must belong to the same space as the target variables.
            Let :math:`x` be input variables and :math:`w` the random model parameters. Then the model is described by
            a function :math:`\mu(w, x)`.
        likelihood_log_variance_model: nn.Module
            A model characterizing the log-variance of a Gaussian likelihood function. The outputs must belong to the
            same space as the target variables. Let :math:`x` be input variables and :math:`w` the random model
            parameters. Then the model is described by a function :math:`\log\sigma^2(w, x)`.
        model_editor : ModelEditor
            A model_editor objects. It takes the forward pass and transforms the outputs.
        seed: int
            A random seed.

        Attributes
        ----------
        model : nn.Module
            See `model` in `Parameters`.
        model_manager : RegressionModelManager
            This object orchestrates the model's forward pass. Given a mean model :math:`\mu(w, x)` and a log-variance
            model :math:`\log\sigma^2`, the model manager concatenates the two into
            :math:`f(w, x)=[\mu(w, x), \log\sigma^2(w, x)]`.
        prob_output_layer : RegressionProbOutputLayer
            This object characterizes the distribution of the target variable given the calibrated outputs. It is
            defined by :math:`p(y|\omega)=\text{Categorical}(p=softmax(\omega))`, where :math:`\omega` denote the
            calibrated outputs and :math:`y` denotes a target variable.
        likelihood : RegressionLikelihood
            The likelihood function. This is defined by
            :math:`p(y|w, \phi, x) = \text{Categorical}(p=\text{softmax}(g(\phi, f(w, x)))`.
        predictive : RegressionPredictive
            This denotes the predictive distribution, that is :math:`p(y|\phi, x, \mathcal{D})`.
        """
        self.model_manager = RegressionModelManager(
            model, likelihood_log_variance_model, model_editor=model_editor
        )
        self.prob_output_layer = RegressionProbOutputLayer()
        self.likelihood = RegressionLikelihood(
            model_manager=self.model_manager,
            prob_output_layer=self.prob_output_layer,
            output_calib_manager=None,
        )
        self.predictive = RegressionPredictive(likelihood=self.likelihood)
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
        """
        Calibrate the calibration regressor.

        Parameters
        ----------
        calib_data_loader : DataLoader
            A calibration data loader.
        val_data_loader : DataLoader
            A validation data loader.
        loss_fn: Callable[[Outputs, Targets], jnp.ndarray]
            The loss function to use for calibration.
        config : Config
            An object to configure the calibration.

        Returns
        -------
        Status
            A calibration status object. It provides information about the calibration.
        """
        self._check_output_dim(calib_data_loader)
        if val_data_loader is not None:
            self._check_output_dim(val_data_loader)
        return self._calibrate(
            calib_data_loader=calib_data_loader,
            uncertainty_fn=(
                config.monitor.uncertainty_fn
                if config.monitor.uncertainty_fn is not None
                else self.prob_output_layer.variance
            ),
            val_data_loader=val_data_loader,
            loss_fn=loss_fn,
            config=config,
        )
