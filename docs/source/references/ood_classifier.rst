.. _ood_detection:

Out-Of-Distribution (OOD) detection
==================
Starting from a trained a neural classifier, it's possible to fit one of the models below
to help distinguish between in-distribution and out of distribution inputs.

.. autoclass:: fortuna.ood_detection.mahalanobis.MalahanobisOODClassifier

.. autoclass:: fortuna.ood_detection.ddu.DeepDeterministicUncertaintyOODClassifier
