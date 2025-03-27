import string
from copy import deepcopy
from datetime import datetime
import random
import numpy as np


class INDIVIDUAL():
    ''' individual class that contains brain and body '''

    def __init__(self, body, brain):
        ''' initialize the individual class with the given body and brain

        Parameters
        ----------
        body: BODY class instance or list of BODY class instances
            defines the shape and muscle properties of the robot

        brain: BRAIN class instance
            defines the controller of the robot

        '''
        # ids
        self.parent_id = ''
        self.parent_fitness = 0.0
        self.parent_bin = None
        self.self_id = ''.join(random.sample(string.ascii_uppercase, k=5))
        self.self_id += '_' + datetime.now().strftime('%H%M%S%f')
        # attributes
        self.fitness = None
        self.fitness_std = None
        self.fitnesses = None
        self.age = 0
        self.self_bin = None
        # initialize main components
        self.body = body
        self.brain = brain

    def _mutate(self, body=None, brain=None):
        # handle ids
        self.parent_id = self.self_id
        self.parent_fitness = self.fitness
        self.parent_bin = self.self_bin
        self.self_id = ''.join(random.sample(string.ascii_uppercase, k=5))
        self.self_id += '_' + datetime.now().strftime('%H%M%S%f')
        self.fitness = None
        self.fitness_std = None
        self.fitnesses = None
        self.self_bin = None
        if body is None and brain is None:
            rand_int = np.random.randint(0, 100)
            if rand_int < 50:
                body = True
                brain = False
            else:
                body = False
                brain = True
        if body:
            self.self_id = 'BODY_' + self.self_id
            self.body.mutate()
        else:
            self.self_id = 'BRAIN_' + self.self_id
            self.brain.mutate()
        return 

    def produce_offspring(self, body=None, brain=None):
        '''
        produce an offspring from the current individual
        '''
        assert not (body == True and brain == True), 'assuming only one of the body or brain is mutated'
        while True:
            offspring = deepcopy(self)
            offspring._mutate(body=body, brain=brain)
            if offspring.is_valid():
                break
        return offspring

    def is_valid(self):
        '''
        check if the individual is valid
        '''
        if self.body.is_valid():
            return True
        else:
            return False

    def __deepcopy__(self, memo):
        """Override deepcopy to apply to class level attributes"""
        new = self.__class__(body=self.body, brain=self.brain)
        new.__dict__.update(deepcopy(self.__dict__, memo))
        return new

