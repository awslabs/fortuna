Posterior
===================
The :ref:`posterior <posterior>` distribution of the model parameters given the training data and the
calibration parameters. We support several posterior approximations:

.. toctree::
   :maxdepth: 1

   map
   advi
   deep_ensemble
   laplace
   swag
   sngp

.. _posterior:

.. autoclass:: fortuna.prob_model.posterior.base.Posterior
    :no-inherited-members:
    :exclude-members: state
    :members: fit, sample, load_state, save_state

.. autoclass:: fortuna.prob_model.posterior.base.PosteriorApproximator

.. autoclass:: fortuna.prob_model.posterior.state.PosteriorState
    :no-inherited-members:
    :exclude-members: params, mutable, calib_params, calib_mutable, replace, encoded_name



