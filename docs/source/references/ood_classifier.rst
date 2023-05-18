.. _ood_detection:

Out-Of-Distribution (OOD) detection
==================
Starting from a trained a neural softmax classifier, it's possible to fit one of the models below
to help distinguish between in-distribution and out of distribution inputs.

All the classes below are abstract and in order to be used the ``apply`` method has to be defined.

.. autoclass:: fortuna.ood_detection.mahalanobis.MalahanobisClassifierABC

.. autoclass:: fortuna.ood_detection.ddu.DeepDeterministicUncertaintyABC
