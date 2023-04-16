Posterior fitting configuration
===============================
This section describes :class:`~fortuna.prob_model.fit_config.base.FitConfig`,
an object that configures the posterior fitting process. It is made of several objects:

- :class:`~fortuna.prob_model.fit_config.optimizer.FitOptimizer`: to configure the optimization process;
- :class:`~fortuna.prob_model.fit_config.checkpointer.FitCheckpointer`: to save and restore checkpoints;
- :class:`~fortuna.prob_model.fit_config.monitor.FitMonitor`: to monitor the process and trigger early stopping;
- :class:`~fortuna.prob_model.fit_config.processor.FitProcessor`: to decide how and where the computation is processed.
- List[:class:`~fortuna.prob_model.callbacks.base.Callback`]: to allow the user to perform custom actions at different stages of the training process.

.. _fit_config:

.. autoclass:: fortuna.prob_model.fit_config.base.FitConfig

.. autoclass:: fortuna.prob_model.fit_config.optimizer.FitOptimizer

.. autoclass:: fortuna.prob_model.fit_config.monitor.FitMonitor

.. autoclass:: fortuna.prob_model.fit_config.checkpointer.FitCheckpointer

.. autoclass:: fortuna.prob_model.fit_config.processor.FitProcessor
