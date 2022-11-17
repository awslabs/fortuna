Maximum-A-Posteriori (MAP)
--------------------------

.. autoclass:: fortuna.prob_model.posterior.map.map_approximator.MAPPosteriorApproximator

.. autoclass:: fortuna.prob_model.posterior.map.map_posterior.MAPPosterior
    :show-inheritance:
    :no-inherited-members:
    :exclude-members: state
    :members: fit, sample, load_state, save_state

.. autoclass:: fortuna.prob_model.posterior.map.map_state.MAPState
    :show-inheritance:
    :no-inherited-members:
    :inherited-members: init, init_from_dict
    :exclude-members: params, mutable, calib_params, calib_mutable, replace, create, apply_gradients, encoded_name
    :no-undoc-members:
    :no-special-members:
