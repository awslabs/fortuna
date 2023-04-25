.. _output_calibrator:

Output calibrator
==================
The output calibration calibrates the model outputs. We explicitly support a
:ref:`temperature scaling output calibrator for classification <output_calibrator_classification>`,
and a :ref:`temperature scaling output calibrator for regression <output_calibrator_regression>`.

Alternatively, you can bring in your own output calibrator by overwriting :mod:`~flax.linen.Module`.

.. _output_calibrator_classification:

.. automodule:: fortuna.output_calibrator.classification
    :no-inherited-members:

.. _output_calibrator_regression:

.. automodule:: fortuna.output_calibrator.regression
    :no-inherited-members:

.. autoclass:: fortuna.output_calib_model.state.OutputCalibState
    :no-inherited-members:
    :exclude-members: params, mutable, encoded_name, replace

