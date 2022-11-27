Calibration configuration
=========================
This section describes :class:`~fortuna.calib_model.calib_config.base.CalibConfig`,
an object that configures the calibration process of the probabilistic model. It is made of several objects:

- :class:`~fortuna.calib_model.calib_config.optimizer.CalibOptimizer`: to configure the optimization process;

- :class:`~fortuna.calib_model.calib_config.checkpointer.CalibCheckpointer`: to save and restore checkpoints;

- :class:`~fortuna.calib_model.calib_config.monitor.CalibMonitor`: to monitor the process and trigger early stopping;

- :class:`~fortuna.calib_model.calib_config.processor.CalibProcessor`: to decide how and where the computation is processed.

.. _calib_model_calib_config:

.. autoclass:: fortuna.calib_model.calib_config.base.CalibConfig

.. _calib_model_calib_optimizer:

.. autoclass:: fortuna.calib_model.calib_config.optimizer.CalibOptimizer

.. _calib_model_calib_checkpointer:

.. autoclass:: fortuna.calib_model.calib_config.checkpointer.CalibCheckpointer

.. _calib_model_calib_monitor:

.. autoclass:: fortuna.calib_model.calib_config.monitor.CalibMonitor

.. _calib_model_calib_processor:

.. autoclass:: fortuna.calib_model.calib_config.processor.CalibProcessor
