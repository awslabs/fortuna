Calibration configuration
=========================
This section describes :class:`~fortuna.prob_model.calib_config.base.CalibConfig`,
an object that configures the calibration process. It is made of several objects:
- :class:`~fortuna.calib_config.optimizer.CalibOptimizer`: to configure the optimization process;
- :class:`~fortuna.calib_config.checkpointer.CalibCheckpointer`: to save and restore checkpoints;
- :class:`~fortuna.calib_config.optimizer.CalibMonitor`: to monitor the process and trigger early stopping;
- :class:`~fortuna.calib_config.optimizer.CalibProcessor`: to decide how and where the computation is processed.

.. _calib_config:

.. autoclass:: fortuna.calib_config.base.CalibConfig
