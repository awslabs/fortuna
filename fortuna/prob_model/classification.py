import importlib
import logging
from typing import (
    Dict,
    Optional,
    Type,
)

import flax.linen as nn

from fortuna.data.loader import DataLoader
from fortuna.likelihood.classification import ClassificationLikelihood
from fortuna.model.model_manager.classification import (
    ClassificationModelManager,
    SNGPClassificationModelManager,
)
from fortuna.model.model_manager.name_to_model_manager import (
    ClassificationModelManagers,
)
from fortuna.model_editor.base import ModelEditor
from fortuna.output_calibrator.classification import ClassificationTemperatureScaler
from fortuna.output_calibrator.output_calib_manager.base import OutputCalibManager
from fortuna.prob_model.base import ProbModel
from fortuna.prob_model.calib_config.base import CalibConfig
from fortuna.prob_model.fit_config.base import FitConfig
from fortuna.prob_model.joint.base import Joint
from fortuna.prob_model.posterior.base import PosteriorApproximator
from fortuna.prob_model.posterior.posterior_approximations import (
    PosteriorApproximations,
)
from fortuna.prob_model.posterior.swag.swag_approximator import (
    SWAGPosteriorApproximator,
)
from fortuna.prob_model.predictive.classification import ClassificationPredictive
from fortuna.prob_model.prior import IsotropicGaussianPrior
from fortuna.prob_model.prior.base import Prior
from fortuna.prob_output_layer.classification import (
    ClassificationMaskedProbOutputLayer,
    ClassificationProbOutputLayer,
)
from fortuna.typing import Status
from fortuna.utils.data import (
    get_input_shape,
    get_inputs_from_shape,
)


class ProbClassifier(ProbModel):
    def __init__(
        self,
        model: nn.Module,
        prior: Prior = IsotropicGaussianPrior(),
        posterior_approximator: PosteriorApproximator = SWAGPosteriorApproximator(),
        output_calibrator: Optional[nn.Module] = ClassificationTemperatureScaler(),
        model_editor: Optional[ModelEditor] = None,
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
        model_editor : ModelEditor
            A model_editor objects. It takes the forward pass and transforms the outputs.
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
            by :math:`p(y|o)=\text{Categorical}(y|p=softmax(o))`,
            where :math:`o` denotes the calibrated outputs and :math:`y` denotes a target variable.
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

        self.output_calib_manager = OutputCalibManager(
            output_calibrator=output_calibrator
        )
        self.prob_output_layer = self._get_prob_output_layer(model)

        model_manager_cls = getattr(
            ClassificationModelManagers, posterior_approximator.__str__()
        ).value
        self.model_manager = self._get_model_manager(
            model, model_editor, model_manager_cls, posterior_approximator
        )

        self.likelihood = ClassificationLikelihood(
            self.model_manager, self.prob_output_layer, self.output_calib_manager
        )
        self.joint = Joint(self.prior, self.likelihood)

        self.posterior = getattr(
            PosteriorApproximations, posterior_approximator.__str__()
        ).value(joint=self.joint, posterior_approximator=posterior_approximator)
        self.predictive = ClassificationPredictive(self.posterior)

        super().__init__(seed=seed)

    def _get_prob_output_layer(self, model: nn.Module) -> ClassificationProbOutputLayer:
        try:
            # import modules if available
            transformers_flax_auto_module = importlib.import_module(
                "transformers.models.auto.modeling_flax_auto"
            )
            FLAX_MODEL_FOR_MASKED_LM_MAPPING_NAMES = list(
                getattr(
                    transformers_flax_auto_module,
                    "FLAX_MODEL_FOR_MASKED_LM_MAPPING_NAMES",
                ).values()
            )
            if str(model.__class__.__name__) in FLAX_MODEL_FOR_MASKED_LM_MAPPING_NAMES:
                prob_output_layer = ClassificationMaskedProbOutputLayer()
            else:
                prob_output_layer = ClassificationProbOutputLayer()
        except ModuleNotFoundError:
            prob_output_layer = ClassificationProbOutputLayer()
        return prob_output_layer

    def _get_model_manager(
        self,
        model: nn.Module,
        model_editor: ModelEditor,
        model_manager_cls: Type,
        posterior_approximator: PosteriorApproximator,
    ) -> ClassificationModelManager:
        try:
            # import modules if available
            transformers_module = importlib.import_module("transformers")
            fortuna_transformers_classification_module = importlib.import_module(
                "fortuna.model.model_manager.transformers.classification"
            )
            # import relevant classes
            FlaxPreTrainedModel = getattr(transformers_module, "FlaxPreTrainedModel")
            SNGPHuggingFaceClassificationModelManager = getattr(
                fortuna_transformers_classification_module,
                "SNGPHuggingFaceClassificationModelManager",
            )
            HuggingFaceClassificationModelManager = getattr(
                fortuna_transformers_classification_module,
                "HuggingFaceClassificationModelManager",
            )
            # load model manager
            if (
                isinstance(model, FlaxPreTrainedModel)
                and model_manager_cls == SNGPClassificationModelManager
            ):
                model_manager = SNGPHuggingFaceClassificationModelManager(
                    model=model,
                    model_editor=model_editor,
                    **posterior_approximator.posterior_method_kwargs,
                )
            elif isinstance(model, FlaxPreTrainedModel):
                model_manager = HuggingFaceClassificationModelManager(
                    model, model_editor=model_editor
                )
            else:
                model_manager = model_manager_cls(
                    model=model,
                    model_editor=model_editor,
                    **posterior_approximator.posterior_method_kwargs,
                )
        except ModuleNotFoundError as e:
            logging.warning(
                "No module named 'transformer' is installed. "
                "If you are not working with models from the `transformers` library ignore this warning, otherwise "
                "install the optional 'transformers' dependency of Fortuna using poetry. You can do so by entering: "
                "`poetry install --extras 'transformers'`."
            )
            model_manager = model_manager_cls(
                model=model,
                model_editor=model_editor,
                **posterior_approximator.posterior_method_kwargs,
            )
        return model_manager

    def _check_output_dim(self, data_loader: DataLoader):
        if data_loader.size == 0:
            raise ValueError(
                """`data_loader` is either empty or incorrectly constructed."""
            )
        output_dim = data_loader.num_unique_labels
        for x, y in data_loader:
            input_shape = get_input_shape(x)
            break
        s = self.joint.init(input_shape)
        inputs = get_inputs_from_shape(input_shape)
        outputs = self.model_manager.apply(
            params=s.params, inputs=inputs, mutable=s.mutable
        )
        model_output_dim = (
            outputs[0].shape[-1]
            if isinstance(outputs, (list, tuple))
            else outputs.shape[-1]
        )
        if model_output_dim != output_dim:
            raise ValueError(
                f"""The outputs dimension of `model` must correspond to the number of different classes
            in the target variables of `_data_loader`. However, {model_output_dim} and {output_dim} were found,
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
