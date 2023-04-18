from typing import Dict, Optional

import flax.linen as nn
import numpy as np

from fortuna.data.loader import DataLoader
from fortuna.model.model_manager.classification import \
    ClassificationModelManager
from fortuna.output_calibrator.classification import \
    ClassificationTemperatureScaler
from fortuna.output_calibrator.output_calib_manager.base import \
    OutputCalibManager
from fortuna.prob_model.base import ProbModel
from fortuna.prob_model.calib_config.base import CalibConfig
from fortuna.prob_model.fit_config.base import FitConfig
from fortuna.prob_model.joint.base import Joint
from fortuna.prob_model.likelihood.classification import \
    ClassificationLikelihood
from fortuna.prob_model.posterior.base import PosteriorApproximator
from fortuna.prob_model.posterior.posterior_approximations import \
    PosteriorApproximations
from fortuna.prob_model.posterior.swag.swag_approximator import \
    SWAGPosteriorApproximator
from fortuna.prob_model.predictive.classification import \
    ClassificationPredictive
from fortuna.prob_model.prior import IsotropicGaussianPrior
from fortuna.prob_model.prior.base import Prior
from fortuna.prob_output_layer.classification import \
    ClassificationProbOutputLayer
from fortuna.typing import Status


class ProbClassifier(ProbModel):
    def __init__(
        self,
        model: nn.Module,
        prior: Prior = IsotropicGaussianPrior(),
        posterior_approximator: PosteriorApproximator = SWAGPosteriorApproximator(),
        output_calibrator: Optional[nn.Module] = ClassificationTemperatureScaler(),
        seed: int = 0,
    ):
        r"""
        A probabilistic classifier class.

        Parameters
        ----------
        model : nn.Module
            A model describing the deterministic relation between inputs and outputs. The outputs must correspond to
            the logits of a softmax probability vector. The output dimension must be the same as the number of classes.
            Let :math:`x` be input variables and :math:`w` the random model parameters. Then the model is described by
            a function :math:`f(w, x)`, where each component of :math:`f` corresponds to one of the classes.
        prior : Prior
            A prior distribution object. The default is an isotropic standard Gaussian. Let :math:`w` be the random
            model parameters. Then the prior is defined by a distribution :math:`p(w)`.
        posterior_approximator : PosteriorApproximator
            A posterior approximation method. The default method is SWAG.
        output_calibrator : Optional[nn.Module]
            An output calibrator object. The default is temperature scaling for classification, which rescales the
            logits with a scalar temperature parameter. Given outputs :math:`o` of the model manager, the output
            calibrator is described by a function :math:`g(\phi, o)`, where `phi` are deterministic
            calibration parameters.
        seed: int
            A random seed.

        Attributes
        ----------
        model : nn.Module
            See `model` in `Parameters`.
        model_manager : ClassificationModelManager
            This object orchestrates the model's forward pass.
        output_calibrator : nn.Module
            See `output_calibrator` in `Parameters`.
        prob_output_layer : ClassificationProbOutputLayer
            This object characterizes the distribution of target variable given the calibrated outputs. It is defined
            by :math:`p(y|\omega)=\text{Categorical}(y|p=softmax(o))`,
            where :math:`o` denote the outputs and :math:`y` denotes a target variable.
        likelihood : ClassificationLikelihood
            The likelihood function. This is defined by
            :math:`p(y|w, \phi, x) = \text{Categorical}(y|p=\text{softmax}(g(\phi, f(w, x)))`.
        prior : Prior
            See `prior` in `Parameters`.
        joint : Joint
            This object describes the joint distribution of the target variables and the random parameters
            given the input variables and the calibration parameters, that is :math:`p(y, w|x, \phi)`.
        posterior_approximator : PosteriorApproximator
            See `posterior_approximator` in `Parameters`.
        posterior : Posterior
            This is the posterior approximation of the random parameters given the training data and the
            calibration parameters, that is :math:`p(w|\mathcal{D}, \phi)`, where :math:`\mathcal{D}` denotes the
            training data set and :math:`\phi` the calibration parameters.
        predictive : ClassificationPredictive
            This denotes the predictive distribution, that is :math:`p(y|\phi, x, \mathcal{D})`. Its statistics are
            approximated via a Monte Carlo approach by sampling from the posterior approximation.
        """
        self.model = model
        self.prior = prior
        self.output_calibrator = output_calibrator

        self.model_manager = ClassificationModelManager(model)
        self.output_calib_manager = OutputCalibManager(
            output_calibrator=output_calibrator
        )
        self.prob_output_layer = ClassificationProbOutputLayer()

        self.likelihood = ClassificationLikelihood(
            self.model_manager, self.prob_output_layer, self.output_calib_manager
        )
        self.joint = Joint(self.prior, self.likelihood)

        self.posterior = getattr(
            PosteriorApproximations, posterior_approximator.__str__()
        ).value(joint=self.joint, posterior_approximator=posterior_approximator)
        self.predictive = ClassificationPredictive(self.posterior)

        super().__init__(seed=seed)

    def _check_output_dim(self, data_loader: DataLoader):
        if data_loader.size == 0:
            raise ValueError(
                """`data_loader` is either empty or incorrectly constructed."""
            )
        output_dim = len(np.unique(data_loader.to_array_targets()))
        for x, y in data_loader:
            input_shape = x.shape[1:]
            break
        s = self.joint.init(input_shape)
        outputs = self.model_manager.apply(
            params=s.params, inputs=np.zeros((1,) + input_shape), mutable=s.mutable
        )
        if outputs.shape[1] != output_dim:
            raise ValueError(
                f"""The outputs dimension of `model` must correspond to the number of different classes
            in the target variables of `_data_loader`. However, {outputs.shape[1]} and {output_dim} were found,
            respectively."""
            )

    def train(
        self,
        train_data_loader: DataLoader,
        val_data_loader: Optional[DataLoader] = None,
        calib_data_loader: Optional[DataLoader] = None,
        fit_config: FitConfig = FitConfig(),
        calib_config: CalibConfig = CalibConfig(),
        **fit_kwargs,
    ) -> Dict[str, Status]:
        self._check_output_dim(train_data_loader)
        return super().train(
            train_data_loader,
            val_data_loader,
            calib_data_loader,
            fit_config,
            calib_config,
            **fit_kwargs,
        )

    def calibrate(
        self,
        calib_data_loader: DataLoader,
        val_data_loader: Optional[DataLoader] = None,
        calib_config: CalibConfig = CalibConfig(),
    ) -> Status:
        """
        Calibrate the probabilistic classifier.

        Parameters
        ----------
        calib_data_loader : DataLoader
            A calibration data loader.
        val_data_loader : DataLoader
            A validation data loader.
        calib_config : CalibConfig
            An object to configure the calibration.

        Returns
        -------
        Status
            A calibration status object. It provides information about the calibration.
        """
        self._check_output_dim(calib_data_loader)
        if val_data_loader is not None:
            self._check_output_dim(val_data_loader)
        return super()._calibrate(
            uncertainty_fn=calib_config.monitor.uncertainty_fn
            if calib_config.monitor.uncertainty_fn is not None
            else self.prob_output_layer.mean,
            calib_data_loader=calib_data_loader,
            val_data_loader=val_data_loader,
            calib_config=calib_config,
        )
