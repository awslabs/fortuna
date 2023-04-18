Calibration configuration
=========================
This section describes :class:`~fortuna.prob_model.calib_config.base.CalibConfig`,
an object that configures the calibration process of the probabilistic model. It is made of several objects:

- :class:`~fortuna.prob_model.calib_config.optimizer.CalibOptimizer`: to configure the optimization process;

- :class:`~fortuna.prob_model.calib_config.checkpointer.CalibCheckpointer`: to save and restore checkpoints;

- :class:`~fortuna.prob_model.calib_config.monitor.CalibMonitor`: to monitor the process and trigger early stopping;

- :class:`~fortuna.prob_model.calib_config.processor.CalibProcessor`: to decide how and where the computation is processed.

.. _prob_calib_config:

.. autoclass:: fortuna.prob_model.calib_config.base.CalibConfig

.. autoclass:: fortuna.prob_model.calib_config.optimizer.CalibOptimizer

.. autoclass:: fortuna.prob_model.calib_config.checkpointer.CalibCheckpointer

.. autoclass:: fortuna.prob_model.calib_config.monitor.CalibMonitor

.. autoclass:: fortuna.prob_model.calib_config.processor.CalibProcessor
