import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import numpy as np
from copy import deepcopy
import _pickle as pickle

from networks import NeuralNetwork, NeuralNetworkBig

class BRAIN(object):
    def __init__(self, name, args):
        self.name = name
        self.args = args

    def mutate(self):
        raise NotImplementedError

    @staticmethod
    def name(self):
        return self.name

    def __deepcopy__(self, memo):
        """Override deepcopy to apply to class level attributes"""
        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(deepcopy(self.__dict__, memo))
        return new

    def is_valid(self):
        raise NotImplementedError

    def extract_brain(self):
        raise NotImplementedError

    def get_action(self, observation):
        # add small noise to the observation
        observation += np.random.normal(0, 0.01, observation.shape)
        # get the actions
        actions = self.model.forward(torch.from_numpy(observation).double())
        # turn the actions into numpy array
        actions = actions.detach().numpy()
        # add small noise to the actions
        actions += np.random.normal(0, 0.01, actions.shape)
        return actions

    def update_model(self):
        if type(self.weights) != torch.Tensor:
            self.weights = torch.from_numpy(self.weights).double()
        vector_to_parameters(self.weights, self.model.parameters())
        self.model.double()

    def set_model(self, model):
        # set requires_grad to False for all parameters
        for p in model.parameters():
            p.requires_grad = False
        self.model = model
        self.weights = parameters_to_vector(self.model.parameters())
        self.model.double()
        self.model.eval()


class CENTRALIZED(BRAIN):
    '''
    '''
    def __init__(self, args):
        BRAIN.__init__(self, "CENTRALIZED", args)
        self.mu, self.sigma = 0, 0.1
        # input size depends on bounding box
        input_size = 0
        if self.args.observe_structure:
            input_size += 5 * self.args.bounding_box[0] * self.args.bounding_box[1]
        if self.args.observe_voxel_speed:
            input_size += 2 * self.args.bounding_box[0] * self.args.bounding_box[1]
        if self.args.observe_voxel_volume:
            input_size += 1 * self.args.bounding_box[0] * self.args.bounding_box[1]
        output_size = self.args.bounding_box[0] * self.args.bounding_box[1]

        if self.args.controller == 'CENTRALIZED':
            self.model = NeuralNetwork(input_size, output_size)
        elif self.args.controller == 'CENTRALIZED_BIG':
            self.model = NeuralNetworkBig(input_size, output_size)
        else:
            raise ValueError('controller not recognized')

        for p in self.model.parameters():
            p.requires_grad = False
        self.weights = parameters_to_vector(self.model.parameters())
        self.model.double()
        self.model.eval()

    def mutate(self):
        noise_weights = np.random.normal(self.mu, self.sigma,
                                         self.weights.shape)
        self.weights += noise_weights
        self.update_model()
        return noise_weights

    def extract_brain(self, path_to_pkl):
        # path should be a .pkl file
        with open(path_to_pkl, 'rb') as f:
            population = pickle.load(f)
        return population[0].brain.model

    def is_valid(self):
        return True




if __name__=="__main__":
    from collections import namedtuple
    kwargs = {'observe_structure': True,
            'observe_voxel_volume': True,
            'observe_voxel_speed': True,
              'observation_range': 1}
    # create args object
    args = namedtuple("args", kwargs.keys())(*kwargs.values())
    print(args)






