from __future__ import division

import random
import numpy as np
import ipyparallel as ipp
import matplotlib.pyplot as plt

from functools import partial
from qinfer.perf_testing import perf_test_multiple, apply_serial
from qinfer import distributions

class hyperheuristic_optimiser(object):
    '''
        A generic hyper-heuristic optimiser class that is inherited by the other optimisation functions.

        :param function fitnessfunction: Function to evalute the fitness of a given dataset.
        :param np.ndarray params: The list of parameters that are being searched over.
        :param function boundaryconditions: Function to constrain points within some boundary regime.
    '''

    def __init__(
                self,
                param_names,
                n_trials,
                model, n_particles, prior,
                n_exp, heuristic,
                boundary_conditions=None,
                experiment_fitness=lambda performance: performance['loss'][:,-1].mean(axis=0),
                apply=apply_serial,
                true_model=None, true_prior=None,
                tskmon_client=None,
                allow_failures=False,
                extra_updater_args=None,
                progressbar=None,
                heuristic_args=None,
                experiment_flag=True
                ):
        self._param_names = param_names
        self._n_free_params = len(param_names)
        self._boundary_conditions = boundary_conditions
        self._n_trials = n_trials #Number of trials
        self._n_particles = n_particles #Number of particles in each trial
        self._n_exp = n_exp # Number of experiments in each trial
        self._model = model # The model used in the experiment
        self._prior = prior # The prior used in the experiment analysis
        if heuristic_args is not None:
            heuristic = partial(heuristic, other_fields=heuristic_args)
        self._heuristic = partial(partial, heuristic) # The heuristic being used
        self._experiment_fitness = experiment_fitness # The evaluation of the heuristic, defaults to the loss
        self._apply = apply
        self._true_model = true_model
        self._true_prior = true_prior
        self._tskmon_client = tskmon_client
        self._allow_failures = allow_failures
        self._extra_updater_args = extra_updater_args
        self._progressbar = progressbar
        self._fitness_dt = None
        self._fitness = None
        self._experiment_flag = experiment_flag

    def fitness_function(self, params):
        if self._experiment_flag:
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
        else: # Allows overriding of the default qinfer experiment simulation with another function
            return self._experiment_fitness(params)
        

    def parrallel(self):
        raise NotImplementedError("This optimiser does not have parrallel support. To resolve this issue, level an appropriate criticism at the developer.")


class particle_swarm_optimiser(hyperheuristic_optimiser):
    """
        A particle swarm optimisation based hyperheuristic
    """

    def __call__(self,
        n_pso_iterations=50,
        n_pso_particles=60,
        initial_position_distribution=None,
        initial_velocity_distribution=None,
        omega_v=0.35, 
        phi_p=0.25, 
        phi_g=0.5
        ):
        self._fitness_dt = np.dtype([
            ('params', np.float64, self._n_free_params),
            ('velocities', np.float64, self._n_free_params),
            ('fitness', np.float64)])
        self._fitness = np.empty([n_pso_iterations, n_pso_particles], dtype=self._fitness_dt)
        local_attractors = np.empty([n_pso_particles], dtype=self._fitness_dt)
        global_attractor = np.empty([1], dtype=self._fitness_dt)

        if initial_position_distribution is None:
            initial_position_distribution = distributions.UniformDistribution(np.array([[ 0, 1]] * self._n_free_params));
            
        if initial_velocity_distribution is None:
            initial_velocity_distribution = distributions.UniformDistribution(np.array([[-1, 1]] * self._n_free_params))
        
        # Initial particle positions
        self._fitness[0]['params'] = initial_position_distribution.sample(n_pso_particles)

        # Apply the boundary conditions if any exist
        if self._boundary_conditions is not None:
            self._fitness[itr]['params'] = self._boundary_conditions(self._fitness[itr]['params'])

        # Calculate the initial particle fitnesses
        self._fitness[0]['fitness'] = self.evaluate_fitness(self._fitness[0]['params'])

        # Calculate the positions of the attractors
        local_attractors = self._fitness[0]
        local_attractors, global_attractor = self.update_attractors(self._fitness[0], local_attractors, global_attractor)

        # Initial particle velocities
        self._fitness[0]['velocities'] = initial_velocity_distribution.sample(n_pso_particles)
        self._fitness[0]['velocities'] = self.update_velocities(
                                        self._fitness[0]['params'], 
                                        self._fitness[0]['velocities'], 
                                        local_attractors['params'],
                                        global_attractor['params'],
                                        omega_v, phi_p, phi_g)

        for itr in range(1, n_pso_iterations):
            #Update the particle positions
            self._fitness[itr]['params'] = self.update_positions(
                self._fitness[itr - 1]['params'], 
                self._fitness[itr - 1]['velocities'])

            # Apply the boundary conditions if any exist
            if self._boundary_conditions is not None:
                self._fitness[itr]['params'] = self._boundary_conditions(self._fitness[itr]['params'])

            # Recalculate the fitness function
            self._fitness[itr]['fitness'] = self.evaluate_fitness(
                self._fitness[itr]['params'])

            # Find the new attractors
            local_attractors, global_attractor = self.update_attractors(self._fitness[itr], local_attractors, global_attractor)

            # Update the velocities
            self._fitness[itr]['velocities'] = self.update_velocities(
                self._fitness[itr]['params'], 
                self._fitness[itr - 1]['velocities'], 
                local_attractors['params'],
                global_attractor['params'],
                omega_v, phi_p, phi_g)

        return global_attractor

    def evaluate_fitness(self, particles):
        fitness = np.empty([len(particles)], dtype=np.float64)
        for idx, particle in enumerate(particles):
            fitness[idx] = self.fitness_function(particle)
        return fitness

    def update_positions(self, positions, velocities):
        updated = positions + velocities
        return updated

    def update_velocities(self, positions, velocities, local_attractors, global_attractor, omega_v, phi_p, phi_g):
        random_p = np.random.random_sample(positions.shape)
        random_g = np.random.random_sample(positions.shape)
        updated = omega_v * velocities + phi_p * random_p * (local_attractors - positions) + phi_g * random_g * (global_attractor - positions) 
        return updated

    def update_attractors(self, particles, local_attractors, global_attractor):
        for idx, particle in enumerate(particles):
            if particle['fitness'] < local_attractors[idx]['fitness']:
                local_attractors[idx] = particle
        global_attractor = local_attractors[np.argmin(local_attractors['fitness'])]
        return local_attractors, global_attractor

#TODO: Debug and test multiple

	def multiple(self,
		n_pso_iterations=50,
		n_pso_particles=60,
		initial_position_distribution=distributions.UniformDistribution(np.array([[ 0, 1]] * self._n_free_params)),
		initial_velocity_distribution=distributions.UniformDistribution(np.array([[-1, 1]] * self._n_free_params)),
		omega_v=0.35, 
		phi_p=0.25, 
		phi_g=0.5,
		verbose=False,
		tskmon_client=None,
		allow_failures=False
		):
	    self._fitness_dt = np.dtype([
			('params', float64, self._n_free_params),
			('velocities', float64, self._n_free_params),
			('fitness', np.float64)])
		self._fitness = np.empty([n_pso_iterations, n_pso_particles], dtype=self._fitness_dt)
		local_attractors = np.empty([n_pso_particles], dtype=self._fitness_dt)
		global_attractor = np.empty([1], dtype=self._fitness_dt)

		# Initial particle positions
		self._fitness[0]['params'] = initial_position_distribution.sample(n_pso_particles)

		# Calculate the initial particle fitnesses
		self._fitness[0]['fitness'] = self.evaluate_fitness(self._fitness[0])

		# Calculate the positions of the attractors
		local_attractors = self._fitness[0]
		local_attractors, global_attractor = update_attractors(self._fitness[0], local_attractors, global_attractor)

		# Initial particle velocities
		self._fitness[0]['velocities'] = initial_velocity_distribution.sample(n_pso_particles)
		self._fitness[0]['velocities'] = update_velocities(self._fitness[itr], omega_v, phi_p, phi_g)

		task = None
    	thread = None
    	wake_event = None
    	prog = None
    	multiple_fn = partial(fitness_function)

		for itr in range(1, n_pso_iterations):
		
			#Update the particle positions
			self._fitness[itr]['params'] = update_positions(
				self._fitness[itr - 1]['params'], 
				self._fitness[itr - 1]['velocities'])

			# Apply the boundary conditions if any exist
			if self._boundary_conditions is not None:
				self._fitness[itr]['params'] = self._boundary_conditions(self._fitness[itr]['params'])

			if tskmon_client is not None:
				try:
					task = tskmon_client.new_task(
						description='QInfer Hyper-heuristic Evaluation',
						status='Evaluating Fitness Function'
						)
					wake_event = threading.Event()
		            thread = WebProgressThread(task, wake_event)
		            thread.daemon = True
		            thread.start()
		        except Exception as ex:
	            	print('Failed to start tskmon task: ', ex)

	            try:
					# Recalculate the fitness function
					results = [self._apply(multiple_fn) for particle in self._fitness[itr]['params']]

					for idx, result in enumerate(results):
						try:
							self._fitness[itr,:]['fitness'] = result.get()

				finally:
					if task is not None:
						try:
			                thread.done = True
			                wake_event.set()
			                task.delete()
			                # Try and join for 1s. If nothing happens, we
			                # raise and move on.
			                thread.join(1)
			                if thread.is_alive():
			                    print("Thread didn't die. This is a bug.")
			            except Exception as ex:
			                print("Exception cleaning up tskmon task.", ex)

			# Find the new attractors
			local_attractors, global_attractor = update_attractors(self._fitness[itr], local_attractors, global_attractor)

			# Update the velocities
			self._fitness[itr]['velocities'] = update_velocities(
				self._fitness[itr]['params'], 
				self._fitness[itr - 1]['velocities'], 
				local_attractors['params'],
				global_attractor['params']
				omega_v, phi_p, phi_g)

		return global_attractor