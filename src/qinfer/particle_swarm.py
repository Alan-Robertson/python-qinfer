from __future__ import division
import random
import numpy as np

from functools import partial

def first(el):
	return el[0]

min_first = partial(min, key=first)

class ParticleSwarmUpdater(object):
	def __init__(self, fitness_function, omega_v=0.6, phi_p=0.3, phi_g=0.1):
	# Tuning Parameters of the Particle Swarm #
		self._fitness_function = fitness_function
		self._omega_v = omega_v
		self._phi_p = phi_p
		self._phi_g = phi_g
		self._p_best = None;
		self._p_best_val = None;
		self._g_best = None;

	## METHODS ##
	def __call__(self, points, velocities):

		# For no initial velocities
		if velocities is None:
			velocities = 2 * np.random.random_sample(points.shape) - 1

		# For no initial best points in particle history
		if self._p_best is None:
			self._p_best = points

		# Associate values with these points
		if self._p_best_val is None:
			self._p_best_val = list(map(self._fitness_function, self._p_best))

		# For no initial best point in swarm history
		if self._g_best is None:
			val, self._g_best = min_first((value, point) for value, point in zip(self._p_best_val, self._p_best))

		vals = np.empty(len(points))
		
		# Update the points in the swarm, this can be parralellised
		# for i in xrange(len(points)):
		for idx, (point, velocity) in enumerate(zip(points, velocities)):
			points[idx], velocities[idx], vals[idx], self._p_best[idx], self._p_best_val[idx] = self.pointupdate(point, velocity, self._p_best[idx], self._g_best)

		val, self._g_best = min_first((value, point) for value, point in zip(self._p_best_val, self._p_best))
		#val, g_best = min((self._p_best_val[i], self._p_best[i]) for i in xrange(len(self._p_best))) 

		return points, velocities, vals

	def pointupdate(self, point, velocity, p_best, g_best):		
		# Random values
		r_p = np.random.random_sample(point.shape)
		r_g = np.random.random_sample(point.shape)

		# Update the velocities
		velocity = self._omega_v * velocity + self._phi_p * r_p * (p_best - point) + self._phi_g * r_g * (g_best - point)

		# Update the points
		point = point + velocity

		# Update the best in path and best in group points
		P = (point, p_best)
		vals = (self._fitness_function(point), self._fitness_function(p_best))

		# Find new p_best
		p_val, p_best = min_first(zip(vals, P))

		return point, velocity, vals[1], p_best, p_val

	# Updating the fitness function
	def fitnessupdate(self, fitness_function):
		self._fitness_function = fitness_function