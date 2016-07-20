from __future__ import division

import random
import numpy as np
import ipyparallel as ipp

from functools import partial
from .particle_swarm import ParticleSwarmUpdater

#Interface for the Particle Swarm Updater
class particle_swarm_optimiser(object):

	def __init__(
			self, 
			FITNESS_FUNCTION,
			PARAMS
			): 
		self._PARAMS = PARAMS # Parameter names of the parameters of the heuristic
		self._FITNESS_FUNCTION = FITNESS_FUNCTION #Evaluates the fitness of a performance dataset
		self._point_history = None # History of all points in the optimisation
		self._val_history = None # Heuristic evaluation of the value of each point in the optimisation

	def __call__(self, 
		N_PSO_ITERATIONS=50,
		N_PSO_PARTICLES=60,
		dist_mean=0, dist_scale=1,
		omega_v=0.2, phi_p=0.4, phi_g=0.4,
		client=None):

		self._point_history = np.empty((N_PSO_ITERATIONS+1, N_PSO_PARTICLES, len(self._PARAMS)))
		self._val_history = np.empty((N_PSO_ITERATIONS+1, N_PSO_PARTICLES)) 

		pso = ParticleSwarmUpdater(self._FITNESS_FUNCTION, omega_v, phi_p, phi_g)
		points = np.random.random((N_PSO_PARTICLES, len(self._PARAMS))) * dist_scale + dist_mean
		self._point_history[0] = points

		points, velocities, vals = pso(points, None)

		for idx in xrange(N_PSO_ITERATIONS):
		    print '%d Percent Complete' %(idx//N_PSO_ITERATIONS)
		    points, velocities, vals = pso(points, velocities)
		    self._point_history[idx+1] = points
		    self._val_history[idx+1] = vals
	    
		return pso._g_best, min(pso._p_best_val)

#class particle_swarm_annealing_optimiser(object):


    
