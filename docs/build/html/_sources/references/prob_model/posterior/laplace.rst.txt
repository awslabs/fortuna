Laplace approximation
---------------------

.. autoclass:: fortuna.prob_model.posterior.laplace.laplace_approximator.LaplacePosteriorApproximator

.. autoclass:: fortuna.prob_model.posterior.laplace.laplace_posterior.LaplacePosterior
    :show-inheritance:
    :no-inherited-members:
    :exclude-members: state
    :members: fit, sample, load_state, save_state

.. autoclass:: fortuna.prob_model.posterior.laplace.laplace_state.LaplaceState
    :show-inheritance:
    :no-inherited-members:
    :inherited-members: init, init_from_dict
    :members: convert_from_map_state
    :exclude-members: params, mutable, calib_params, calib_mutable, replace, apply_gradients, encoded_name, create
    :no-undoc-members:
    :no-special-members:
