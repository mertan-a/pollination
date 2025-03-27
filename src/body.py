import numpy as np
from copy import deepcopy

from cppn import mutate_network, calc_outputs, CPPN
from evogym import is_connected
from utils import make_one_shape_only
import settings

class BODY(object):
    def __init__(self, type):
        self.type = type

    def mutate(self):
        raise NotImplementedError

    def to_phenotype(self):
        raise NotImplementedError

    def count_existing_voxels(self):
        voxel_data = self.to_phenotype()
        return np.sum(voxel_data > 0)

    def count_active_voxels(self):
        voxel_data = self.to_phenotype()
        return np.sum(voxel_data == 3) + np.sum(voxel_data == 4)

    def is_valid(self):
        voxel_data = self.to_phenotype()
        if np.isnan(voxel_data).any():
            return False
        if np.sum(voxel_data == 3) + np.sum(voxel_data == 4) < 3:
            return False
        if is_connected(voxel_data) == False:
            return False
        return True

class CPPN_BODY(BODY):
    def __init__(self, args):
        BODY.__init__(self, "cppn")
        self.args = args
        self.output_node_names = ["0", "1", "2", "3", "4"]
        self.network = CPPN(output_node_names=self.output_node_names)
        self.orig_size_xyz = [1,self.args.bounding_box[0],self.args.bounding_box[1]]
        self.mutate()

    def to_phenotype(self):
        calc_outputs(
            self.network, self.orig_size_xyz, self)
        voxel_data = np.stack(
            [self.network.graph.nodes[node_name]["state"] for node_name in self.output_node_names], axis=-1)
        voxel_data = np.squeeze(voxel_data)
        voxel_data = np.argmax(voxel_data, axis=-1)
        return voxel_data

    def mutate(self):
        initial_network = deepcopy(self.network)
        initial_structure = self.to_phenotype()
        if settings.VERBOSE:
            print()
            print("-----------------")
            print("cppn body mutation")
        while self.is_valid() == False or np.all(self.to_phenotype() == initial_structure):
            self.network = deepcopy(initial_network)
            mutate_network(self.network)
        if settings.VERBOSE:
            print("done with cppn body mutation")
            print("initial structure")
            print(initial_structure)
            print("mutated structure")
            print(self.to_phenotype())
            print("-----------------")

class BASIC_BODY(BODY):
    def __init__(self, args):
        BODY.__init__(self, "evolvable")
        self.args = args
        self.structure = np.zeros((self.args.bounding_box[0], self.args.bounding_box[1]))
        self.random_body()

    def to_phenotype(self):
        return self.structure

    def mutate(self):
        if settings.VERBOSE:
            print()
            print("-----------------")
            print("basic body mutation")
        initial_structure = deepcopy(self.structure)
        while self.is_valid() == False or np.all(self.structure == initial_structure):
            self.structure = deepcopy(initial_structure)
            for i in range(self.structure.shape[0]):
                for j in range(self.structure.shape[1]):
                    mutation_prob = 0.2
                    if np.random.random() < mutation_prob:
                        if self.structure[i,j] == 0:
                            self.structure[i,j] = np.random.choice([1,2,3,4])
                        elif self.structure[i][j] == 1:
                            self.structure[i][j] = np.random.choice([0,2,3,4])
                        elif self.structure[i][j] == 2:
                            self.structure[i][j] = np.random.choice([0,1,3,4])
                        elif self.structure[i][j] == 3:
                            self.structure[i][j] = np.random.choice([0,1,2,4])
                        else:
                            self.structure[i][j] = np.random.choice([0,1,2,3])
        if settings.VERBOSE:
            print("done with basic body mutation")
            print("initial structure")
            print(initial_structure)
            print("mutated structure")
            print(self.structure)
            print("-----------------")

    def random_body(self):
        valid = False
        while not valid:
            self.structure = np.random.randint(0, 5, (self.args.bounding_box[0], self.args.bounding_box[1]))
            mask = make_one_shape_only(self.structure)
            self.structure = self.structure * mask
            valid = self.is_valid()

if __name__ == "__main__":
    body = CPPN_BODY(bounding_box=(1, 10, 10,))
    print(body.network.graph)
    #import networkx as nx
    #import matplotlib.pyplot as plt
    #nx.draw(body.network.graph, with_labels=True)
    #plt.show()
    while True:
        body.mutate()
        print(body.network.graph)
        print(body.is_valid())
        phenotyped_body = body.to_phenotype()
        print(phenotyped_body)
        # check the number of unique materials
        if len(np.unique(phenotyped_body)) < 7:
            continue
        input()




