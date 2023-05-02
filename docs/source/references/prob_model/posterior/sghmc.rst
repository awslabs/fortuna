Stochastic Gradient Hamiltonian Monte Carlo (SGHMC)
------------------------------------------------------

.. autoclass:: fortuna.prob_model.posterior.sgmcmc.sghmc.sghmc_approximator.SGHMCPosteriorApproximator

.. autoclass:: fortuna.prob_model.posterior.sgmcmc.sghmc.sghmc_posterior.SGHMCPosterior
    :show-inheritance:
    :no-inherited-members:
    :exclude-members: state
    :members: fit, sample, load_state, save_state

.. autoclass:: fortuna.prob_model.posterior.sgmcmc.sghmc.sghmc_state.SGHMCState
    :show-inheritance:
    :no-inherited-members:
    :inherited-members: init, init_from_dict
    :members: convert_from_map_state
    :exclude-members: params, mutable, calib_params, calib_mutable, replace, apply_gradients, encoded_name, create
    :no-undoc-members:
    :no-special-members:


Step schedules
==============

Fortuna supports various step schedulers for SG-MCMC
algorithms. :class:`~fortuna.rob_model.posterior.sgmcmc.sgmcmc_step_schedule.StepSchedule`
is a function that takes step count as an input and returns `float` step
size as an output.

.. automodule:: fortuna.prob_model.posterior.sgmcmc.sgmcmc_step_schedule


Preconditioners
===============

Fortuna provides implementations of preconditioners to improve samplers efficacy.

.. automodule:: fortuna.prob_model.posterior.sgmcmc.sgmcmc_preconditioner
    :exclude-members: Preconditioner, PreconditionerState, RMSPropPreconditionerState, IdentityPreconditionerState


Diagnostics
===========

The library includes toolings necessary for diagnostics of the convergence of
SG-MCMC sampling algorithms.

.. automodule:: fortuna.prob_model.posterior.sgmcmc.sgmcmc_diagnostic
