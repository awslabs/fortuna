Training Callbacks
===============================
This section describes :class:`~fortuna.prob_model.callbacks.base.Callback` that allow users to add custom actions at different stages of the training loop.

In order to train a :class:`~fortuna.prob_model.base.ProbModel` using callbacks the user have to:

- Define its own callbacks by subclassing :class:`~fortuna.prob_model.callbacks.base.Callback` and override the methods of interest.
- When calling the train method for a :class:`~fortuna.prob_model.base.ProbModel` instance, add a list of callbacks containing the ones previously defined when initializing :class:`~fortuna.prob_model.fit_config.base.FitConfig`.

.. _callback:

.. autoclass:: fortuna.prob_model.callbacks.base.Callback
