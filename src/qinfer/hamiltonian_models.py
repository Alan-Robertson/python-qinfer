
#Features
from __future__ import division
from __future__ import absolute_import

# Exports
__all__ = [
    'JaynesCummingsModel',
]

#Imports
import random
import numpy as np
from functools import partial
from .abstract_model import Model

# Renamed in near future
class JaynesCummingsModel(Model):

    def __init__(self, min_freq=0, Q=np.array([1,0,0])):
        super(JaynesCummingsModel, self).__init__()
        self._min_freq = min_freq
        self._Q = Q

    ## PROPERTIES ##
    
    @property
    def n_modelparams(self):
        return 3
    
    @property
    def modelparam_names(self):
        return ['modefrequency','couplingstrength','relaxationtime']
    
    @property
    def expparams_dtype(self):
        return [('qubitfrequency', 'float'), ('time', 'float')]
    
    @property
    def is_n_outcomes_constant(self):
        """
        Returns ``True`` if and only if the number of outcomes for each
        experiment is independent of the experiment being performed.
        
        This property is assumed by inference engines to be constant for
        the lifetime of a Model instance.
        """
        return True
    
    ## METHODS ##
    
    def are_models_valid(self, modelparams):
        return np.all(modelparams > self._min_freq, axis=1)
    
    def n_outcomes(self, expparams):
        """
        Returns an array of dtype ``uint`` describing the number of outcomes
        for each experiment specified by ``expparams``.
        
        :param numpy.ndarray expparams: Array of experimental parameters. This
            array must be of dtype agreeing with the ``expparams_dtype``
            property.
        """
        return 2
    
    def likelihood(self, outcomes, modelparams, expparams):
        # By calling the superclass implementation, we can consolidate
        # call counting there.
        super(JaynesCummingsModel, self).likelihood(
            outcomes, modelparams, expparams
        )
        # Possibly add a second axis to modelparams.
        if len(modelparams.shape) == 1:
            modelparams = modelparams[..., np.newaxis]
        
        # Each of these has shape (n_models, 1)
        mode_freq, coupling_strength, relaxation_time = (modelparams.T)[:, :, np.newaxis]
        
        # Allocating first serves to make sure that a shape mismatch later
        # will cause an error.
        pr0 = np.zeros((modelparams.shape[0], expparams.shape[0]))
        
        detuningfrequency = mode_freq - expparams['qubitfrequency']
        rabifrequency = np.sqrt(detuningfrequency ** 2 + 4 * coupling_strength ** 2) 
        
        pr0[:, :] = ( 1 
                - ((rabifrequency + detuningfrequency)/(2 * rabifrequency)) ** 2 * 
                np.exp((-1 * (rabifrequency + detuningfrequency) * expparams['time'])/
                (2 * rabifrequency * relaxation_time))
                - ((rabifrequency - detuningfrequency)/(2 * rabifrequency)) ** 2 * 
                np.exp((-1 * (rabifrequency - detuningfrequency) * expparams['time'])/
                (2 * rabifrequency * relaxation_time))
                - (2 * coupling_strength ** 2)/(rabifrequency ** 2) 
                * np.exp((-expparams['time'])/(2 * relaxation_time)) 
                * np.cos(rabifrequency * expparams['time'])
               )
        
        # Now we concatenate over outcomes.
        return Model.pr0_to_likelihood_array(outcomes, pr0)



        # Renamed in near future
class PettaModel(Model):
    def __init__(self, min_freq=0, Q=np.array([1,0,0])):
        super(PettaModel, self).__init__()
        self._min_freq = min_freq
        self._Q = Q

    ## PROPERTIES ##
    
    @property
    def n_modelparams(self):
        return 1
    
    @property
    def modelparam_names(self):
        return ['detuning']
    
    @property
    def expparams_dtype(self):
        return [('exchangetime', 'float')]
    
    @property
    def is_n_outcomes_constant(self):
        """
        Returns ``True`` if and only if the number of outcomes for each
        experiment is independent of the experiment being performed.
        
        This property is assumed by inference engines to be constant for
        the lifetime of a Model instance.
        """
        return True
    
    ## METHODS ##
    
    def are_models_valid(self, modelparams):
        return np.all(modelparams > self._min_freq, axis=1)
    
    def n_outcomes(self, expparams):
        """
        Returns an array of dtype ``uint`` describing the number of outcomes
        for each experiment specified by ``expparams``.
        
        :param numpy.ndarray expparams: Array of experimental parameters. This
            array must be of dtype agreeing with the ``expparams_dtype``
            property.
        """
        return 2
    
    def likelihood(self, outcomes, modelparams, expparams):
        # By calling the superclass implementation, we can consolidate
        # call counting there.
        super(PettaModel, self).likelihood(
            outcomes, modelparams, expparams
        )
        # Possibly add a second axis to modelparams.
        if len(modelparams.shape) == 1:
            modelparams = modelparams[..., np.newaxis]
        
        # Each of these has shape (n_models, 1)
        detuning = (modelparams.T)[:, :, np.newaxis]
        exchangetime = expparams
        
        # Allocating first serves to make sure that a shape mismatch later
        # will cause an error.
        pr0 = np.zeros((modelparams.shape[0], expparams.shape[0]))
        
        pr0[:, :] = ((3 - np.cos(detuning * exchangetime))/4)
        
        # Now we concatenate over outcomes.
        return Model.pr0_to_likelihood_array(outcomes, pr0)