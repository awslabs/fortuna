Calibration configuration
=========================
This section describes :class:`~fortuna.output_calib_model.config.base.Config`,
an object that configures the calibration process of the probabilistic model. It is made of several objects:

- :class:`~fortuna.output_calib_model.config.optimizer.Optimizer`: to configure the optimization process;

- :class:`~fortuna.output_calib_model.config.checkpointer.Checkpointer`: to save and restore checkpoints;

- :class:`~fortuna.output_calib_model.config.monitor.Monitor`: to monitor the process and trigger early stopping;

- :class:`~fortuna.output_calib_model.config.processor.Processor`: to decide how and where the computation is processed.

.. _output_calib_model_config:

.. autoclass:: fortuna.output_calib_model.config.base.Config

.. _output_calib_model_calib_optimizer:

.. autoclass:: fortuna.output_calib_model.config.optimizer.Optimizer

.. _output_calib_model_calib_checkpointer:

.. autoclass:: fortuna.output_calib_model.config.checkpointer.Checkpointer

.. _output_calib_model_calib_monitor:

.. autoclass:: fortuna.output_calib_model.config.monitor.Monitor

.. _output_calib_model_calib_processor:

.. autoclass:: fortuna.output_calib_model.config.processor.Processor
