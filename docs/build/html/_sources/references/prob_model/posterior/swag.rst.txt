SWAG
-----

.. autoclass:: fortuna.prob_model.posterior.swag.swag_approximator.SWAGPosteriorApproximator

.. autoclass:: fortuna.prob_model.posterior.swag.swag_posterior.SWAGPosterior
    :show-inheritance:
    :no-inherited-members:
    :exclude-members: state
    :members: fit, sample, load_state, save_state

.. autoclass:: fortuna.prob_model.posterior.swag.swag_state.SWAGState
    :show-inheritance:
    :no-inherited-members:
    :inherited-members: init, init_from_dict
    :members: convert_from_map_state, update
    :exclude-members: params, mutable, calib_params, calib_mutable, replace, mean, std, dev, create, apply_gradients,
                      encoded_name
    :no-undoc-members:
    :no-special-members:
