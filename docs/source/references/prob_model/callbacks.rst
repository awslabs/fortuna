Training Callbacks
===============================
This section describes :class:`~fortuna.prob_model.fit_config.callback.FitCallback`,
which allows users to add custom actions at different stages of the training loop.
Callbacks can be used while training a :class:`~fortuna.prob_model.base.ProbModel`.

To use callbacks the user has to:

- Define their own callbacks by subclassing :class:`~fortuna.prob_model.fit_config.callback.FitCallback` and override the methods of interest.
- When calling the train method of a :class:`~fortuna.calib_model.base.ProbModel` instance,
  add a list of callbacks to the configuration object :class:`~fortuna.prob_model.fit_config.base.FitConfig`.

The following example outlines the usage of :class:`~fortuna.prob_model.fit_config.callback.FitCallback`.
It assumes that the user already obtained an instance of :class:`~fortuna.prob_model.base.ProbModel`:

.. code-block:: python

    from jax.flatten_util import ravel_pytree
    import optax

    from fortuna.training.train_state import TrainState
    from fortuna.prob_model.fit_config import FitConfig, FitMonitor, FitOptimizer, FitCallback
    from fortuna.metric.classification import accuracy

    # Define custom callback
    class CountParamsCallback(FitCallback):
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

.. autoclass:: fortuna.training.callback.Callback
