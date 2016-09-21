from __future__ import division

import random
import numpy as np
import ipyparallel as ipp
import matplotlib.pyplot as plt

from functools import partial
from .particle_swarm import ParticleSwarmUpdater, min_first


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

    def plot_freq_dist(self, x=0, y=1, size=15, cmap='hot'):
        N_PSO_ITERATIONS, N_PSO_PARTICLES, N_PARAMS = self._point_history.shape
        point_filter = np.concatenate([self._point_history[i,:,[x,y]] for i in range(N_PSO_ITERATIONS)], axis=1)
        point_vals = np.concatenate([self._val_history[i] for i in range(N_PSO_ITERATIONS)], axis=0)
        plt.scatter(point_filter[0], point_filter[1], c=point_vals, s=size, cmap=cmap)


    def plot_path(self, x=0, y=1, p_plot=[0], size=15, cmap='hot', params=None):
        for p in p_plot:
            point_filter = self._point_history[:,p,[x,y]][:-1,:]
            point_vals = self._val_history[:-1,p]

            plt.plot(point_filter[:,0], point_filter[:,1])
            plt.scatter(point_filter[:,0], point_filter[:,1], c=point_vals, s=size,cmap=cmap)

    def plot_opt_history(self, params=None):
        N_PSO_ITERATIONS, N_PSO_PARTICLES, N_PARAMS = self._point_history.shape
        history = np.zeros(N_PSO_ITERATIONS)

        c_best = min(self._val_history[0,:])
        history[0] = c_best

        for i in range(N_PSO_ITERATIONS):
            history[i] = min(c_best, min(self._val_history[i,:]))
            c_best = history[i]

        plt.plot(history)


#Interface for the Particle Swarm Updater
class particle_swarm_optimiser(optimiser):

    def __call__(self, 
        N_PSO_ITERATIONS=50,
        N_PSO_PARTICLES=60,
        dist_mean=0, dist_scale=1,
        omega_v=0.35, phi_p=0.25, phi_g=0.5,
        verbose=False,
        client=None):

        # Initialise the point history and the value history
        if verbose:
            print("Initialising History...")
        self._point_history = np.empty((N_PSO_ITERATIONS+1, N_PSO_PARTICLES, len(self._PARAMS)))
        self._val_history = np.empty((N_PSO_ITERATIONS+1, N_PSO_PARTICLES)) 

        # Initialise the swarm and the points
        if verbose:
            print("Initialising Points and Velocities...")
        pso = ParticleSwarmUpdater(self._FITNESS_FUNCTION, omega_v, phi_p, phi_g)
        points = np.random.random((N_PSO_PARTICLES, len(self._PARAMS))) * dist_scale + dist_mean

        # First run of the swarm, get the velocities from this
        points, velocities, vals = pso(points, None)
        self._point_history[0] = points
        self._val_history[0] = vals

        # Iterate for each 
        if verbose:
            print("Beginning Iterations...")
        for idx in xrange(N_PSO_ITERATIONS):
            if verbose:
                print '%d Percent Complete' %((100*idx)//N_PSO_ITERATIONS)

            points, velocities, vals = pso(points, velocities)
            points = self._BOUNDARY_CONDITIONS(points)
            self._point_history[idx+1] = points
            self._val_history[idx+1] = vals
        
        if verbose:
                print '100 Percent Complete'

        return pso._g_best, pso._g_best_val

class particle_swarm_rejection_annealing_optimiser(optimiser):

    # A particle swarm object that performs annealing during the optimisation,
    # The annealing is set by the cooling rate, or may be set for each parameter individually
    # using the asymmetric cooling rate as a vector with one entry for each parameter

    def __call__(self, 
        N_PSO_ITERATIONS=50,
        N_PSO_PARTICLES=60,
        COOLING_RATE = 0.99,
        dist_mean=0, dist_scale=1,
        omega_v=0.35, phi_p=0.25, phi_g=0.5,
        verbose=False,
        client=None):

    	# Acceptance rate for rejection
        acceptance_rate = 1

        self._point_history = np.empty((N_PSO_ITERATIONS+1, N_PSO_PARTICLES, len(self._PARAMS)))
        self._val_history = np.empty((N_PSO_ITERATIONS+1, N_PSO_PARTICLES)) 

        pso = ParticleSwarmUpdater(self._FITNESS_FUNCTION, omega_v, phi_p, phi_g)
        points = np.random.random((N_PSO_PARTICLES, len(self._PARAMS))) * dist_scale + dist_mean

        points, velocities, vals = pso(points, None)
        self._point_history[0] = points
        self._val_history[0] = vals

        for idx in xrange(N_PSO_ITERATIONS):
            if verbose:
                print '%d Percent Complete' %((100*idx)//N_PSO_ITERATIONS)
            points_u, velocities_u, vals_u = pso(points, velocities)

            for idx_val, _ in enumerate(vals_u):
            	if vals_u[idx_val] * (1 - acceptance_rate) < vals[idx_val]:
            		points[idx_val] = points_u[idx_val]
            		velocities[idx_val] = velocities_u[idx_val]
            		vals[idx_val] = vals_u[idx_val]

            points = self._BOUNDARY_CONDITIONS(points)
            self._point_history[idx+1] = points
            self._val_history[idx+1] = vals

            acceptance_rate*=COOLING_RATE


        return pso._g_best, pso._g_best_val


class particle_swarm_annealing_optimiser(optimiser):
    def __call__(self, 
        N_PSO_ITERATIONS=50,
        N_PSO_PARTICLES=60,
        COOLING_RATE = 0.99,
        ASYM_COOLING_RATE = None,
        dist_mean=0, dist_scale=1,
        omega_v=0.7, phi_p=0.5, phi_g=1,
        verbose=False,
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
            if verbose:
                print '%d Percent Complete' %((100*idx)//N_PSO_ITERATIONS)
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
        N_TEMPER_CATEGORIES=6,
        TEMPER_FREQUENCY=10,
        TEMPER_VALUES=None,
        dist_mean=0, dist_scale=1,
        verbose=False,
        client=None):

        if (TEMPER_VALUES is None):
            TEMPER_VALUES = np.random.random((N_TEMPER_CATEGORIES, 3))
        else:
            N_TEMPER_CATEGORIES=len(TEMPER_VALUES)

        self._point_history = np.empty((N_PSO_ITERATIONS+1, N_PSO_PARTICLES, len(self._PARAMS)))
        self._val_history = np.empty((N_PSO_ITERATIONS+1, N_PSO_PARTICLES)) 

        # Generate an array of different swarms, each with different tempers
        pso = [ParticleSwarmUpdater(self._FITNESS_FUNCTION, temper[0], temper[1], temper[2]) for temper in TEMPER_VALUES]
        temper_list = range(0,N_PSO_PARTICLES)
        np.random.shuffle(temper_list)
        di = int(np.floor(N_PSO_PARTICLES//N_TEMPER_CATEGORIES))
        temper_map = [temper_list[i:i+di] for i in range(0,N_PSO_PARTICLES, di)]
        N_TEMPER_CATEGORIES = len(temper_map)

        # Initialise the points, velocities and values
        points = np.random.random((N_PSO_PARTICLES, len(self._PARAMS))) * dist_scale + dist_mean
        velocities = 2*np.zeros(points.shape) - 1
        vals = np.zeros(N_PSO_PARTICLES)
        self._point_history[0] = points

        # Need to manually handle p_best and g_best as we swap the particles
        p_best = np.zeros(points.shape)
        p_best_val = np.zeros(N_PSO_PARTICLES)
        g_best = 0
        g_best_val = 0
        
        # Initial run
        for p_idx, particle in enumerate(pso):
            points[temper_map[p_idx]], velocities[temper_map[p_idx]], vals[temper_map[p_idx]] = particle(points[temper_map[p_idx]], None)
        self._point_history[0] = points
        self._val_history[0] = vals

        # The particle swarm iterations
        for idx in xrange(N_PSO_ITERATIONS):
            if verbose:
                print '%d Percent Complete' %((100*idx)//N_PSO_ITERATIONS)

            g_best_val, g_best = min_first((min(particle._p_best_val), particle._g_best) for particle in pso)

            # Update the points
            for p_idx, particle in enumerate(pso):
                particle._g_best = g_best #Update g_best of the swarm
                points[temper_map[p_idx]], velocities[temper_map[p_idx]], vals[temper_map[p_idx]] = particle(points[temper_map[p_idx]], velocities[temper_map[p_idx]])
                points[temper_map[p_idx]] = self._BOUNDARY_CONDITIONS(points[temper_map[p_idx]])

            # Save the point history
            self._point_history[idx+1] = points
            self._val_history[idx+1] = vals

            # After the required number of iterations the distribution is tempered
            if (idx % TEMPER_FREQUENCY == 0) and (0 != idx):

                # Save p_best and p_best_val to the correct particles
                for p_idx, particle in enumerate(pso):
                    p_best[temper_map[p_idx]] = particle._p_best
                    p_best_val[temper_map[p_idx]] = particle._p_best_val

                # Shuffle the list and reconstruct the map
                np.random.shuffle(temper_list)
                temper_map = [temper_list[i:i+di] for i in range(0,N_PSO_PARTICLES, di)]

                # Set p_best and p_best_val for the new swarms
                for p_idx, particle in enumerate(pso):
                    particle._p_best = p_best[temper_map[p_idx]]
                    particle._p_best_val = p_best_val[temper_map[p_idx]]

        return g_best, g_best_val

class SPSA_optimiser(optimiser):

    def __call__(self, 
        N_SPSA_ITERATIONS=50,
        N_SPSA_PARTICLES=60,
        dist_mean=0, dist_scale=1,
        A = 0, s = 1/3, t = 1, 
        a = 0.5, b = 0.5,
        verbose=False,
        client=None):

        # SPSA functions
        alpha = lambda k: a / (1 + A + k)**s
        beta  = lambda k: b / (1 + k)**t
        delta = lambda d: (2 * np.round(np.random.random(d))) - 1

        # Initialise the swarm and the points
        self._point_history = np.empty((N_SPSA_ITERATIONS, N_SPSA_PARTICLES, len(self._PARAMS)))
        self._val_history = np.empty((N_SPSA_ITERATIONS, N_SPSA_PARTICLES))

        points = np.random.random((N_SPSA_PARTICLES, len(self._PARAMS))) * dist_scale + dist_mean

        self._point_history[0] = points
        self._val_history[0] = [self._FITNESS_FUNCTION(points[particle]) for particle in xrange(N_SPSA_PARTICLES)]

        for iteration in xrange(1,N_SPSA_ITERATIONS):
            for particle in xrange(N_SPSA_PARTICLES):
                
                # Generate delta
                delta_k = delta(len(self._PARAMS))

                # Calculate the point updates
                points[particle] += (beta(iteration) 
                    * (self._FITNESS_FUNCTION(points[particle] - alpha(iteration)*delta_k) 
                    -  self._FITNESS_FUNCTION(points[particle] + alpha(iteration)*delta_k)) 
                    *  delta_k / (2 * alpha(iteration)))

            # Save the point history
            self._point_history[iteration] = points
            self._val_history[iteration] = [self._FITNESS_FUNCTION(points[particle]) for particle in xrange(N_SPSA_PARTICLES)]

        g_best_val = min(self._val_history[N_SPSA_ITERATIONS-1])

        return g_best_val
            