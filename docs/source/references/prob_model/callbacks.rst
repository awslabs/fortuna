Training Callbacks
===============================
This section describes :class:`~fortuna.prob_model.callbacks.base.Callback` that allows users to add custom actions at different stages of the training loop.

In order to train a :class:`~fortuna.prob_model.base.ProbModel` using callbacks the user has to:

- Define their own callbacks by subclassing :class:`~fortuna.prob_model.callbacks.base.Callback` and override the methods of interest.
- When calling the train method for a :class:`~fortuna.prob_model.base.ProbModel` instance, add a list of callbacks containing the ones previously defined when initializing :class:`~fortuna.prob_model.fit_config.base.FitConfig`.

The following example outlines the usage of :class:`~fortuna.prob_model.callbacks.base.Callback`.
It assumes that the user already obtained an insatnce of :class:`~fortuna.prob_model.base.ProbModel`:

.. code-block:: python

    from jax.flatten_util import ravel_pytree
    import optax

    from fortuna.prob_model.callbacks.base import Callback
    from fortuna.prob_model.fit_config import FitConfig, FitMonitor, FitOptimizer
    from fortuna.metric.classification import accuracy

    # Define custom callback
    class CountParamsCallback(Callback):
        def training_epoch_start(self, state: TrainState) -> TrainState:
            params, unravel = ravel_pytree(state.params)
            logger.info(f"num params: {len(params)}")
            return state

    # Add a list of callbacks containing CountParamsCallback to FitConfig
    status = prob_model.train(
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        calib_data_loader=val_data_loader,
        fit_config=FitConfig(
            optimizer=FitOptimizer(method=optax.adam(1e-4), n_epochs=100),
            callbacks=[
                CountParamsCallback()
            ]
        )
    )


.. _callbacks:

.. autoclass:: fortuna.prob_model.callbacks.base.Callback

.. autoclass:: fortuna.prob_model.callbacks.sngp.ResetCovarianceCallback

