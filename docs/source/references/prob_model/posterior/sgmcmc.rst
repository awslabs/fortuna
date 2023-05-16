Stochastic Gradient Markov Chain Monte Carlo (SG-MCMC)
------------------------------------------------------
SG-MCMC procedures approximate the posterior as a steady-state distribution of
a Monte Carlo Markov chain, that utilizes noisy estimates of the gradient
computed on minibatches of data.

Stochastic Gradient Hamiltonian Monte Carlo (SGHMC)
===================================================

SGHMC `[Chen T. et al., 2014] <http://proceedings.mlr.press/v32/cheni14.pdf>`_
is a popular MCMC algorithm that uses stochastic gradient estimates to scale
to large datasets.

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

Cyclical Stochastic Gradient Langevin Dynamics (CyclicalSGLD)
=============================================================

Cyclical SGLD method `[Zhang R. et al., 2019] <https://openreview.net/pdf?id=rkeS1RVtPS>`_ is a simple and automatic
procedure that adapts the cyclical cosine stepsize schedule, and alternates between
*exploration* and *sampling* stages to better explore the multimodal posteriors for deep neural networks.

.. autoclass:: fortuna.prob_model.posterior.sgmcmc.cyclical_sgld.cyclical_sgld_approximator.CyclicalSGLDPosteriorApproximator

.. autoclass:: fortuna.prob_model.posterior.sgmcmc.cyclical_sgld.cyclical_sgld_posterior.CyclicalSGLDPosterior
    :show-inheritance:
    :no-inherited-members:
    :exclude-members: state
    :members: fit, sample, load_state, save_state

.. autoclass:: fortuna.prob_model.posterior.sgmcmc.cyclical_sgld.cyclical_sgld_state.CyclicalSGLDState
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
algorithms. :class:`~fortuna.prob_model.posterior.sgmcmc.sgmcmc_step_schedule.StepSchedule`
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
SG-MCMC sampling procedures.

.. automodule:: fortuna.prob_model.posterior.sgmcmc.sgmcmc_diagnostic
