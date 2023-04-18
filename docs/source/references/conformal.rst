Conformal prediction
====================
Conformal prediction methods are a type of calibration methods that, starting from uncertainty estimates,
provide *conformal sets*, i.e. rigorous sets of predictions with a user-chosen level of probability.
We support conformal methods for both
:ref:`classification <conformal_classification>`
and :ref:`regression <conformal_regression>`.

.. _conformal_classification:

.. automodule:: fortuna.calibration.conformal.classification.adaptive_prediction

.. automodule:: fortuna.calibration.conformal.classification.simple_prediction

.. automodule:: fortuna.calibration.conformal.classification.adaptive_conformal_classifier

.. automodule:: fortuna.calibration.conformal.classification.batch_mvp

.. _conformal_regression:

.. automodule:: fortuna.calibration.conformal.regression.quantile

.. automodule:: fortuna.calibration.conformal.regression.onedim_uncertainty

.. automodule:: fortuna.calibration.conformal.regression.cvplus

.. automodule:: fortuna.calibration.conformal.regression.jackknifeplus

.. automodule:: fortuna.calibration.conformal.regression.jackknife_minmax

.. automodule:: fortuna.calibration.conformal.regression.enbpi

.. automodule:: fortuna.calibration.conformal.regression.adaptive_conformal_regressor

.. automodule:: fortuna.calibration.conformal.regression.batch_mvp
