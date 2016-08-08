from __future__ import division
from __future__ import absolute_import

from functools import partial
from qinfer.perf_testing import perf_test_multiple, apply_serial
import numpy as np

class heuristic_simulation():

	def __init__(self, 
			n_trials,
			model, n_particles, prior,
			n_exp, heuristic, param_names, 
			experiment_fitness=lambda performance: performance['loss'][:,-1].mean(axis=0),
			apply=apply_serial,
			true_model=None, true_prior=None,
			tskmon_client=None,
        	allow_failures=False,
        	extra_updater_args=None,
        	progressbar=None,
        	heuristic_args=None
		):

		self._n_trials = n_trials #Number of trials
		self._n_particles = n_particles #Number of particles in each trial
		self._n_exp = n_exp # Number of experiments in each trial
		self._model = model # The model used in the experiment
		self._prior = prior # The prior used in the experiment analysis
		self._param_names = param_names # The parameters of the heuristic
		if heuristic_args is not None:
			heuristic = partial(heuristic, other_fields=heuristic_args)
		self._heuristic = partial(partial, heuristic) # The heuristic being used
		self._experiment_fitness = experiment_fitness # The evaluation of the heuristic, defaults to the loss
		self._apply=apply
		self._true_model = true_model
		self._true_prior = true_prior
		self._tskmon_client = tskmon_client
		self._allow_failures = allow_failures
		self._extra_updater_args = extra_updater_args
		self._progressbar = progressbar

	def __call__(self, params):
	    performance = perf_test_multiple(
	        n_trials = self._n_trials, 
	        model = self._model, 
	        n_particles = self._n_particles, 
	        prior = self._prior, 
	        n_exp = self._n_exp,
	        heuristic_class  = self._heuristic(**{
	            name: param
	            for name, param in zip(self._param_names, params)
	        }), 
	        apply = self._apply,
	        true_model = self._true_model, true_prior = self._true_prior,
	        tskmon_client = self._tskmon_client,
        	allow_failures = self._allow_failures,
        	extra_updater_args = self._extra_updater_args,
        	progressbar = self._progressbar
	    )
	    return self._experiment_fitness(performance)