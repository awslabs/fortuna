Conformal prediction
====================
Conformal prediction methods are a type of calibration methods that, starting from uncertainty estimates,
provide *conformal sets*, i.e. rigorous sets of predictions with a user-chosen level of probability.
We support conformal methods for both
:ref:`classification <conformal_classification>`
and :ref:`regression <conformal_regression>`.

.. _conformal_classification:

.. automodule:: fortuna.conformal.classification.adaptive_prediction

.. automodule:: fortuna.conformal.classification.simple_prediction

.. automodule:: fortuna.conformal.classification.adaptive_conformal_classifier

.. automodule:: fortuna.conformal.classification.batch_mvp

.. automodule:: fortuna.conformal.classification.multicalibrator

.. _conformal_regression:

.. automodule:: fortuna.conformal.regression.quantile

.. automodule:: fortuna.conformal.regression.onedim_uncertainty

.. automodule:: fortuna.conformal.regression.cvplus

.. automodule:: fortuna.conformal.regression.jackknifeplus

.. automodule:: fortuna.conformal.regression.jackknife_minmax

.. automodule:: fortuna.conformal.regression.enbpi

.. automodule:: fortuna.conformal.regression.adaptive_conformal_regressor

.. automodule:: fortuna.conformal.regression.batch_mvp

.. automodule:: fortuna.conformal.regression.multicalibrator
