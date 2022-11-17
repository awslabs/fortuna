Posterior fitting configuration
===============================
This section describes :class:`~fortuna.prob_model.fit_config.base.FitConfig`,
an object that configures the posterior fitting process. It is made of several objects:
- :class:`~fortuna.prob_model.fit_config.optimizer.FitOptimizer`: to configure the optimization process;
- :class:`~fortuna.prob_model.fit_config.checkpointer.FitCheckpointer`: to save and restore checkpoints;
- :class:`~fortuna.prob_model.fit_config.optimizer.FitMonitor`: to monitor the process and trigger early stopping;
- :class:`~fortuna.prob_model.fit_config.optimizer.FitProcessor`: to decide how and where the computation is processed.

.. _fit_config:

.. autoclass:: fortuna.prob_model.fit_config.base.FitConfig

.. autoclass:: fortuna.prob_model.fit_config.optimizer.FitOptimizer

.. autoclass:: fortuna.prob_model.fit_config.monitor.FitMonitor

.. autoclass:: fortuna.prob_model.fit_config.checkpointer.FitCheckpointer

.. autoclass:: fortuna.prob_model.fit_config.processor.FitProcessor
