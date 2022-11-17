Save and load checkpoints
=========================
Fortuna lets you save checkpoints while training the probabilistic model,
and restoring them back at a future time.

Saving a checkpoint
-------------------
In order to save checkpoints to a directory :code:`save_checkpoint_dir`,
you just need to communicate this to the
:class:`~fortuna.prob_model.posterior.fit_config.checkpointer.FitCheckpointer` object in
:class:`~fortuna.prob_model.posterior.fit_config.base.FitConfig`, which configures
the posterior fitting process. Given a probabilistic model :code:`prob_model`
and some training data loader :code:`train_data_loader`,
here is a minimal code example.

.. code-block:: python
    :caption: **References:** :meth:`~fortuna.prob_model.base.ProbModel.train`

    status = prob_model.train(
        train_data_loader=train_data_loader,
        fit_config=FitConfig(
            checkpointer=FitCheckpointer(
                save_checkpoint_dir=save_checkpoint_dir
            )
        )
    )

The :class:`~fortuna.prob_model.posterior.fit_config.checkpointer.FitCheckpointer` has several other configuration options,
including how many iterations checkpoints should be saved (:code:`save_every_n_steps`),
how many checkpoints to keep in the directory from the most recent (:code:`keep_top_n_checkpoints`),
and whether the state of the posterior should be saved on disk rather than kept in memory (:code:`save_state`).


Restoring a checkpoint for further training
-------------------------------------------
Suppose that you want to continue training the probabilistic model starting from an existing
checkpoint saved in :code:`restore_checkpoint_path`. The latter can be either a filepath,
or a path to a directory containing valid checkpoints.
Given a probabilistic model :code:`prob_model`
and some training data loader :code:`train_data_loader`,
here is a minimal code example.

.. code-block:: python
    :caption: **References:** :meth:`~fortuna.prob_model.base.ProbModel.train`, :class:`~fortuna.prob_model.posterior.fit_config.base.FitConfig`, :class:`~fortuna.prob_model.posterior.fit_config.checkpointer.FitCheckpointer`

    status = prob_model.train(
        train_data_loader=train_data_loader,
        fit_config=FitConfig(
            checkpointer=FitCheckpointer(
                restore_checkpoint_path=restore_checkpoint_path
            )
        )
    )

Save and load a state of a probabilistic model
------------------------------------------------
Perhaps you have already trained the probabilistic model and saved a checkpoint in :code:`checkpoint_path`,
and at a future time you want to load this as the state of you probabilistic model :code:`prob_model`.
Here is a minimal example to do this.

.. code-block:: python
    :caption: **References:** :meth:`~fortuna.prob_model.base.ProbModel.load_state`

    prob_model.load_state(
        checkpoint_path=checkpoint_path
    )

You might use this for further calibration by invoking the method
:meth:`~fortuna.prob_model.base.ProbModel.calibrate`,
or for making predictions, or more.

If your :code:`prob_model` already contains a state, you can save this very simply in
:code:`checkpoint_path` as follows.

.. code-block:: python
    :caption: **References:** :meth:`~fortuna.prob_model.base.ProbModel.save_state`

    prob_model.save_state(
        checkpoint_path=checkpoint_path
    )


