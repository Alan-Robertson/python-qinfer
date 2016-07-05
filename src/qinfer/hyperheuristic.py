from __future__ import division
from __future__ import absolute_import

from functools import partial
from qinfer.perf_testing import perf_test_multiple, apply_serial
import numpy as np


class HyperHeuristic():

	def __init__(self, 
			n_trials,
			model, n_particles, prior, 
			n_exp, hyper_heuristic, param_names, 
			apply=apply_serial, 
			heuristic='loss'
		):

		self._n_trials = n_trials
		self._n_particles = n_particles
		self._n_exp = n_exp
		self._model = model
		self._prior = prior
		self._param_names = param_names
		self._hyper_heuristic = hyper_heuristic
		self._heuristic = heuristic
		self._apply=apply

	def __call__(self, params):
	    output = perf_test_multiple(
	        self._n_trials, self._model, self._n_particles, self._prior, self._n_exp,
	        self._hyper_heuristic(**{
	            name: param
	            for name, param in zip(self._param_names, params)
	        }), apply=self._apply
	    )
	    return output[self._heuristic][:, -1].mean(axis=0)