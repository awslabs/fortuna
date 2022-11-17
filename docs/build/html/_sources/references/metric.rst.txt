Metric
===========
We support some metrics for both
:ref:`classification <metric_classification>`
and :ref:`regression <metric_regression>`.
Metrics are `NumPy <https://numpy.org/doc/stable/>`__-compatible, therefore feel free to bring your own and
apply them on Fortuna's predictions.

.. _metric_classification:

.. automodule:: fortuna.metric.classification
    :exclude-members: compute_counts_confs_accs

.. _metric_regression:

.. automodule:: fortuna.metric.regression
