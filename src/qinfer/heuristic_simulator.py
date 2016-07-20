from __future__ import division
from __future__ import absolute_import

from functools import partial
from qinfer.perf_testing import perf_test_multiple, apply_serial
import numpy as np

# Rename this class
class heuristic_simulation():

	def __init__(self, 
			n_trials,
			model, n_particles, prior, 
			n_exp, heuristic, param_names, 
			experiment_fitness=lambda performance: performance['loss'][:,-1].mean(axis=0),
			apply=apply_serial
		):

		self._n_trials = n_trials #Number of trials
		self._n_particles = n_particles #Number of particles in each trial
		self._n_exp = n_exp # Number of experiments in each trial
		self._model = model # The model used in the experiment
		self._prior = prior # The prior used in the experiment analysis
		self._param_names = param_names # The parameters of the heuristic
		self._heuristic = partial(partial, heuristic) # The heuristic being used
		self._experiment_fitness = experiment_fitness # The evaluation of the heuristic, defaults to the loss
		self._apply=apply

	def __call__(self, params):
	    performance = perf_test_multiple(
	        self._n_trials, self._model, self._n_particles, self._prior, self._n_exp,
	        self._heuristic(**{
	            name: param
	            for name, param in zip(self._param_names, params)
	        }), apply=self._apply
	    )
	    return self._experiment_fitness(performance)