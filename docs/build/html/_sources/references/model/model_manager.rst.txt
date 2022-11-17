.. _model_manager:

Model manager
=============
The model manager is responsible for the orchestration of the forward pass.
We support a :ref:`classification model manager <model_manager_classification>` for classification
and a :ref:`regression model manager <model_manager_regression>` for regression.

.. _model_manager_classification:

.. automodule:: fortuna.model.model_manager.classification

.. _model_manager_regression:

.. automodule:: fortuna.model.model_manager.regression

.. automodule:: fortuna.model.model_manager.base

.. autoclass:: fortuna.model.model_manager.state.ModelManagerState
    :no-inherited-members:
    :exclude-members: params, mutable
