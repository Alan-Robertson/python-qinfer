from __future__ import division

import random
import numpy as np
import ipyparallel as ipp

from functools import partial
from .particle_swarm import ParticleSwarmUpdater


class optimiser(object):
	def __init__(
				self, 
				FITNESS_FUNCTION,
				PARAMS,
				BOUNDARY_CONDITIONS=lambda points: points
				): 
			self._PARAMS = PARAMS # Parameter names of the parameters of the heuristic
			self._FITNESS_FUNCTION = FITNESS_FUNCTION # Evaluates the fitness of a performance dataset
			self._BOUNDARY_CONDITIONS = BOUNDARY_CONDITIONS # Function that acts on the points at the end of each iteration, defaults to no action
			self._point_history = None # History of all points in the optimisation
			self._val_history = None # Heuristic evaluation of the value of each point in the optimisation


#Interface for the Particle Swarm Updater
class particle_swarm_optimiser(optimiser):

	def __call__(self, 
		N_PSO_ITERATIONS=50,
		N_PSO_PARTICLES=60,
		dist_mean=0, dist_scale=1,
		omega_v=0.2, phi_p=0.4, phi_g=0.4,
		client=None):

		# Initialise the point history and the value history
		self._point_history = np.empty((N_PSO_ITERATIONS+1, N_PSO_PARTICLES, len(self._PARAMS)))
		self._val_history = np.empty((N_PSO_ITERATIONS+1, N_PSO_PARTICLES)) 

		# Initialise the swarm and the points
		pso = ParticleSwarmUpdater(self._FITNESS_FUNCTION, omega_v, phi_p, phi_g)
		points = np.random.random((N_PSO_PARTICLES, len(self._PARAMS))) * dist_scale + dist_mean

		# First run of the swarm, get the velocities from this
		points, velocities, vals = pso(points, None)
		self._point_history[0] = points
		self._val_history[0] = vals

		# Iterate for each 
		for idx in xrange(N_PSO_ITERATIONS):
		    print '%d Percent Complete' %(100*idx//N_PSO_ITERATIONS)
		    points, velocities, vals = pso(points, velocities)
		    points = self._BOUNDARY_CONDITIONS(points)
		    self._point_history[idx+1] = points
		    self._val_history[idx+1] = vals
	    
		return pso._g_best, min(pso._p_best_val)

class particle_swarm_annealing_optimiser(optimiser):

	# A particle swarm object that performs annealing during the optimisation,
	# The annealing is set by the cooling rate, or may be set for each parameter individually
	# using the asymmetric cooling rate as a vector with one entry for each parameter

	def __call__(self, 
		N_PSO_ITERATIONS=50,
		N_PSO_PARTICLES=60,
		COOLING_RATE = 0.99,
		ASYM_COOLING_RATE = None,
		dist_mean=0, dist_scale=1,
		omega_v=0.2, phi_p=0.4, phi_g=0.4,
		client=None):

		if (ASYM_COOLING_RATE is None):
			ASYM_COOLING_RATE = [COOLING_RATE, COOLING_RATE, COOLING_RATE]

		self._point_history = np.empty((N_PSO_ITERATIONS+1, N_PSO_PARTICLES, len(self._PARAMS)))
		self._val_history = np.empty((N_PSO_ITERATIONS+1, N_PSO_PARTICLES)) 

		pso = ParticleSwarmUpdater(self._FITNESS_FUNCTION, omega_v, phi_p, phi_g)
		points = np.random.random((N_PSO_PARTICLES, len(self._PARAMS))) * dist_scale + dist_mean
		self._point_history[0] = points

		points, velocities, vals = pso(points, None)

		for idx in xrange(N_PSO_ITERATIONS):

		    print '%d Percent Complete' %(100*idx//N_PSO_ITERATIONS)
		    points, velocities, vals = pso(points, velocities)
		    points = self._BOUNDARY_CONDITIONS(points)
		    self._point_history[idx+1] = points
		    self._val_history[idx+1] = vals

		    pso._omega_v *= ASYM_COOLING_RATE[0]
	    	pso._phi_p *= ASYM_COOLING_RATE[1]
	    	pso._phi_g *= ASYM_COOLING_RATE[2]


		return pso._g_best, min(pso._p_best_val)
    

class particle_swarm_tempering_optimiser(optimiser):

	def __call__(self, 
		N_PSO_ITERATIONS=50,
		N_PSO_PARTICLES=60,
		TEMPER_CATEGORIES=6,
		TEMPER_FREQUENCY=10,
		TEMPER_VALUES=None,
		dist_mean=0, dist_scale=1,
		client=None):

		if (TEMPER_VALUES is None):
			TEMPER_VALUES = np.random.random((TEMPER_CATEGORIES, 3))

		self._point_history = np.empty((N_PSO_ITERATIONS+1, N_PSO_PARTICLES, len(self._PARAMS)))
		self._val_history = np.empty((N_PSO_ITERATIONS+1, N_PSO_PARTICLES)) 


		

		pso = [ParticleSwarmUpdater(self._FITNESS_FUNCTION, temper[0], temper[1], temper[2]) for temper in TEMPER_VALUES]

		points = np.random.random((N_PSO_PARTICLES, len(self._PARAMS))) * dist_scale + dist_mean
		self._point_history[0] = points

		for idx, particle in enumerate(pso[0:-1]):
			points[idx:idx+1], velocities[idx:idx+1], vals[idx:idx+1] = particle(points[idx:idx+1], None)

		for idx in xrange(N_PSO_ITERATIONS):

			print '%d Percent Complete' %(100*idx//N_PSO_ITERATIONS)
			g_best =  min(particle._g_best for particle in pso)

			for p_idx, particle in enumerate(pso):
				particle._g_best = g_best
				points[p_idx], velocities[p_idx], vals[p_idx] = particle(points[p_idx], velocities[p_idx])
				points[p_idx] = self._BOUNDARY_CONDITIONS(points[p_idx])
				self._point_history[idx+1,p_idx] = points[p_idx]
				self._val_history[idx+1,p_idx] = vals[p_idx]

			if (idx % TEMPER_FREQUENCY == 0):
				np.random.shuffle(TEMPER_VALUES)
			for p_idx, particle in enumerate(pso):
				particle._omega_v = TEMPER_VALUES[p_idx, 0]
				particle._phi_p = TEMPER_VALUES[p_idx,1]
				particle._phi_g = TEMPER_VALUES[p_idx,2]

		return pso._g_best, min(pso._p_best_val)