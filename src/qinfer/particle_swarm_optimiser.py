from __future__ import division

import random
import numpy as np
import ipyparallel as ipp

from functools import partial
from .abstract_model import Model
from .particle_swarm import ParticleSwarmUpdater
from .hyperheuristic import HyperHeuristic
from .abstract_model import Model


#Interface for the Particle Swarm Updater
class particle_swarm_optimiser(object):

	def __init__(
			self, 
			MODEL, PRIOR, 
			HEURISTIC, HYPER_HEURISTIC, HYPER_PARAMS, 
			FITNESS_FUNCTION=lambda performance: performance['loss'][:,-1].mean(axis=0)
			):
		self._MODEL = MODEL # Model class object
		self._PRIOR = PRIOR # Prior class object
		self._HEURISTIC = HEURISTIC # Heuristic Class object
		self._HYPER_HEURISTIC = HYPER_HEURISTIC    
		self._HYPER_PARAMS = HYPER_PARAMS # Parameter names of the parameters of the heuristic
		self._FITNESS_FUNCTION = FITNESS_FUNCTION #Evaluates the fitness of a performance dataset
		self._point_history = None # History of all points in the optimisation
		self._val_history = None # Heuristic evaluation of the value of each point in the optimisation

	def __call__(self, 
		N_PSO_ITERATIONS=50,
		N_PSO_PARTICLES=60, 
		N_PARTICLES=1000, 
		N_TRIALS=100, N_EXP=100,
		dist_mean=0, dist_scale=1,
		omega_v=0.6, phi_p=0.3, phi_g=0.1,
		client=None):

		self._point_history = numpy.empty((N_PSO_ITERATIONS+1, N_PSO_PARTICLES, len(self._HYPER_PARAMS)))
		self._point_val = numpy.empty((N_PSO_ITERATIONS+1, N_PSO_PARTICLES)) 

		if client is None: # No parrallelisation
			fitness = HyperHeuristic(
						N_TRIALS, self._MODEL, 
						N_PARTICLES, self._PRIOR, 
						N_EXP, self._HYPER_HEURISTIC, 
						self._HYPER_PARAMS,
						self._FITNESS_FUNCTION
						)
		else: # Parrallelisation
			lbview = client.load_balanced_view()
			fitness = HyperHeuristic(
						N_TRIALS, self._MODEL, 
						N_PARTICLES, self._PRIOR, 
						N_EXP, self._HYPER_HEURISTIC, 
						self._HYPER_PARAMS, 
						self._FITNESS_FUNCTION,
						lbview.apply
						)

		pso = ParticleSwarmUpdater(fitness, omega_v, phi_p, phi_g)
		points = numpy.random.random((N_PSO_PARTICLES, len(self._HYPER_PARAMS))) * dist_scale + dist_mean
		self._point_history[0] = points

		points, velocities, vals = pso(points, None)

		for idx in xrange(N_PSO_ITERATIONS):
		    print '%d Percent Complete' %(idx//N_PSO_ITERATIONS)
		    points, velocities = pso(test_points, velocities)
	    
		return pso._g_best, min(pso._p_best_val)
    
