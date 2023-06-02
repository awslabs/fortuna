import importlib
import logging
from typing import (
    Callable,
    Optional,
    Type,
)

from flax import linen as nn
import jax.numpy as jnp

from fortuna.calib_model.base import CalibModel
from fortuna.calib_model.config.base import Config
from fortuna.calib_model.predictive.classification import ClassificationPredictive
from fortuna.data import DataLoader
from fortuna.likelihood.classification import ClassificationLikelihood
from fortuna.loss.classification.focal_loss import focal_loss_fn
from fortuna.model.model_manager.classification import ClassificationModelManager
from fortuna.model_editor.base import ModelEditor
from fortuna.prob_output_layer.classification import (
    ClassificationMaskedProbOutputLayer,
    ClassificationProbOutputLayer,
)
from fortuna.typing import (
    Outputs,
    Status,
    Targets,
)
from fortuna.utils.data import get_input_shape


class CalibClassifier(CalibModel):
    def __init__(
        self,
        model: nn.Module,
        model_editor: Optional[ModelEditor] = None,
        seed: int = 0,
    ):
        r"""
        A calibration classifier class.

        Parameters
        ----------
        model : nn.Module
            A model describing the deterministic relation between inputs and outputs. The outputs must correspond to
            the logits of a softmax probability vector. The output dimension must be the same as the number of classes.
            Let :math:`x` be input variables and :math:`w` the random model parameters. Then the model is described by
            a function :math:`f(w, x)`, where each component of :math:`f` corresponds to one of the classes.
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
        prob_output_layer : ClassificationProbOutputLayer
            This object characterizes the distribution of target variable given the outputs. It is defined
            by :math:`p(y|o)=\text{Categorical}(y|p=softmax(o))`,
            where :math:`o` denotes the outputs and :math:`y` denotes a target variable.
        likelihood : ClassificationLikelihood
            The likelihood function. This is defined by
            :math:`p(y|w, \phi, x) = \text{Categorical}(y|p=\text{softmax}(g(\phi, f(w, x)))`.
        predictive : ClassificationPredictive
            This denotes the predictive distribution, that is :math:`p(y|x, \mathcal{D})`.
        """
        self.model_manager = self._get_model_manager(
            model, model_editor, ClassificationModelManager
        )
        self.prob_output_layer = self._get_prob_output_layer(model)
        self.likelihood = ClassificationLikelihood(
            model_manager=self.model_manager,
            prob_output_layer=self.prob_output_layer,
            output_calib_manager=None,
        )
        self.predictive = ClassificationPredictive(likelihood=self.likelihood)
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
        model_editor: Optional[nn.Module],
        model_manager_cls: Type,
    ) -> ClassificationModelManager:
        try:
            # import modules if available
            transformers_module = importlib.import_module("transformers")
            fortuna_transformers_classification_module = importlib.import_module(
                "fortuna.model.model_manager.transformers.classification"
            )
            # import relevant classes
            FlaxPreTrainedModel = getattr(transformers_module, "FlaxPreTrainedModel")
            HuggingFaceClassificationModelManager = getattr(
                fortuna_transformers_classification_module,
                "HuggingFaceClassificationModelManager",
            )
            # load model manager
            if isinstance(model, FlaxPreTrainedModel):
                model_manager = HuggingFaceClassificationModelManager(
                    model, model_editor
                )
            else:
                model_manager = model_manager_cls(model, model_editor)
        except ModuleNotFoundError as e:
            logging.warning(
                "No module named 'transformer' is installed. "
                "If you are not working with models from the `transformers` library ignore this warning, otherwise "
                "please install the optional 'transformers' dependency of fortuna."
                'Using poetry, you can achieve this by entering: `poetry install --extras "transformers"`'
            )
            model_manager = model_manager_cls(model, model_editor)
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
        model_manager_output_dim = self._get_output_dim(input_shape)
        if model_manager_output_dim != output_dim:
            raise ValueError(
                f"""The outputs dimension of `model` must correspond to the number of different classes
            in the target variables of `data_loader`. However, {model_manager_output_dim} and {output_dim} were
            found, respectively."""
            )

    def calibrate(
        self,
        calib_data_loader: DataLoader,
        val_data_loader: Optional[DataLoader] = None,
        loss_fn: Callable[[Outputs, Targets], jnp.ndarray] = focal_loss_fn,
        config: Config = Config(),
    ) -> Status:
        """
        Calibrate the calibration classifier.

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
            uncertainty_fn=config.monitor.uncertainty_fn
            if config.monitor.uncertainty_fn is not None
            else self.prob_output_layer.mean,
            val_data_loader=val_data_loader,
            loss_fn=loss_fn,
            config=config,
        )
