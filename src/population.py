import operator
import numpy as np
import os
import _pickle as pickle

from individual import INDIVIDUAL
from body import CPPN_BODY, BASIC_BODY
from brain import CENTRALIZED
import settings

class ARCHIVE():
    """A population of individuals to be used with MAP-Elites"""

    def __init__(self, args):
        """Initialize a population of individuals.

        Parameters
        ----------
        args : object
            arguments object

        """
        self.args = args
        self.archive = {}
        # archive is assumed to be 2D
        for i in range(0, self.args.archive_bins[0]):
            for j in range(0, self.args.archive_bins[1]):
                self.archive[(i,j)] = None

    def get_random_individual(self):
        valid = False
        while not valid:
            # body
            if self.args.body_representation == 'DIRECT':
                body = BASIC_BODY(self.args)
            elif self.args.body_representation == 'CPPN':
                body = CPPN_BODY(self.args)
            else:
                raise ValueError("Invalid body type")
            # brain
            if self.args.controller == 'CENTRALIZED' or self.args.controller == 'CENTRALIZED_BIG':
                brain = CENTRALIZED(self.args)
            else:
                raise ValueError("Invalid controller type")
            ind = INDIVIDUAL(body=body, brain=brain)
            if ind.is_valid():
                valid = True
        return ind

    def produce_offsprings(self, generation):
        """Produce offspring from the current map."""
        # check if there are any individuals in the map
        if len(self) == 0:
            init_population = []
            while len(init_population) < self.args.nr_parents:
                init_population.append(self.get_random_individual())
            if settings.VERBOSE:
                print("Archive is currently empty. Random individuals are generated.")
                for ind in init_population:
                    print(ind.self_id)
            return init_population
        # choose nr_parents many random keys from the map. make sure that they are not None
        valid_keys = [ k for k in self.archive.keys() if self.archive[k] is not None ]
        nr_valid_keys = len(valid_keys) if len(valid_keys) < self.args.nr_parents else self.args.nr_parents
        random_keys_idx = np.random.choice(len(valid_keys), size=nr_valid_keys, replace=False)
        # produce offsprings
        offsprings = []
        for key_idx in random_keys_idx:
            key = valid_keys[key_idx]
            offspring = self.archive[key].produce_offspring()
            offsprings.append(offspring)
            if settings.VERBOSE:
                print()
                print('-----------------')
                print("Parent selected: ", self.archive[key].self_id)
                print(f"parent's bin: {offspring.parent_bin}")
                print("Offspring produced: ", offspring.self_id)
                print(f"offspring's bin: {self.determine_bins(offspring)}")
            if offspring.self_id.startswith('BODY'):
                print("Body mutation")
                print("Parent's body:")
                print(self.archive[key].body.to_phenotype())
                print("Offspring's body:")
                print(offspring.body.to_phenotype())
                print('-----------------')
        return offsprings

    def __iter__(self):
        """Iterate over the individuals. Use the expression 'for n in population'."""
        individuals = [ ind for ind in self.archive.values() if ind is not None ]
        return iter(individuals)

    def __contains__(self, n):
        """Return True if n is a SoftBot in the population, False otherwise. Use the expression 'n in population'."""
        individuals = [ ind.self_id for ind in self.archive.values() if ind is not None ]
        try:
            return n.self_id in individuals
        except AttributeError:
            return False

    def __len__(self):
        """Return the number of individuals in the population. Use the expression 'len(population)'."""
        individuals = [ ind for ind in self.archive.values() if ind is not None ]
        return len(individuals)

    def __getitem__(self, x):
        """Return individual n.  Use the expression 'population[n]'."""
        return self.archive[x]

    def get_individual_with_id(self, id):
        """Return the individual with the given id."""
        individuals = [ ind for ind in self.archive.values() if ind is not None and ind.self_id == id ]
        if len(individuals) == 0:
            return None
        return individuals[0]

    def get_best_individual(self):
        """Return the best individual in the population."""
        individuals = [ ind for ind in self.archive.values() if ind is not None ]
        return max(individuals, key=operator.attrgetter('fitness'))

    def get_best_fitness(self):
        """Return the best fitness in the population."""
        individuals = [ ind for ind in self.archive.values() if ind is not None ]
        return max(individuals, key=operator.attrgetter('fitness')).fitness

    def get_individuals(self):
        """Return the individuals in the population."""
        individuals = [ ind for ind in self.archive.values() if ind is not None ]
        return individuals

    def get_individuals_ids(self):
        """Return the ids of the individuals in the population."""
        individuals = [ ind.self_id for ind in self.archive.values() if ind is not None ]
        return individuals

    def update_archive(self, population, gen):
        """Update the map with the given population."""
        #### RECORD KEEPING
        ## offspring / parent fitness
        ratio_offspring_parent = [] # overall
        ratio_offspring_parent_brain_mutation = [] # brain mutations only
        ratio_offspring_parent_body_mutation = [] # body mutations only
        ratio_offspring_parent_brain_mutation_distilled = [] # brain mutations only, child of distilled parent
        ratio_offspring_parent_body_mutation_distilled = [] # body mutations only, child of distilled parent
        ratio_offspring_parent_brain_mutation_regular = [] # brain mutations only, child of regular parent
        ratio_offspring_parent_body_mutation_regular = [] # body mutations only, child of regular parent
        for offspring in population:
            if offspring.parent_id == '':
                continue
            ratio = offspring.fitness / offspring.parent_fitness
            if offspring.fitness is not None:
                ratio_offspring_parent.append(ratio)
            if offspring.self_id.startswith('BRAIN') and offspring.fitness is not None:
                ratio_offspring_parent_brain_mutation.append(ratio)
                if "DISTILLED" in offspring.parent_id:
                    ratio_offspring_parent_brain_mutation_distilled.append(ratio)
                else:
                    ratio_offspring_parent_brain_mutation_regular.append(ratio)
            if offspring.self_id.startswith('BODY') and offspring.fitness is not None:
                ratio_offspring_parent_body_mutation.append(ratio)
                if "DISTILLED" in offspring.parent_id:
                    ratio_offspring_parent_body_mutation_distilled.append(ratio)
                else:
                    ratio_offspring_parent_body_mutation_regular.append(ratio)

        # over time (every generation)
        if os.path.exists(os.path.join(self.args.rundir, 'evolution_summary', 'mutation_fitness_changes_overtime.pkl')):
            with open(os.path.join(self.args.rundir, 'evolution_summary', 'mutation_fitness_changes_overtime.pkl'), 'rb') as f:
                fitness_changes_overtime = pickle.load(f)
            fitness_changes_overtime[gen] = np.array(ratio_offspring_parent)
            with open(os.path.join(self.args.rundir, 'evolution_summary', 'mutation_fitness_changes_overtime.pkl'), 'wb') as f:
                pickle.dump(fitness_changes_overtime, f)
        else:
            fitness_changes_overtime = {}
            fitness_changes_overtime[gen] = np.array(ratio_offspring_parent)
            with open(os.path.join(self.args.rundir, 'evolution_summary', 'mutation_fitness_changes_overtime.pkl'), 'wb') as f:
                pickle.dump(fitness_changes_overtime, f)
        # brain mutations only, over time (every generation)
        if os.path.exists(os.path.join(self.args.rundir, 'evolution_summary', 'mutation_fitness_changes_brain_overtime.pkl')):
            with open(os.path.join(self.args.rundir, 'evolution_summary', 'mutation_fitness_changes_brain_overtime.pkl'), 'rb') as f:
                fitness_changes_brain_overtime = pickle.load(f)
            fitness_changes_brain_overtime[gen] = np.array(ratio_offspring_parent_brain_mutation)
            with open(os.path.join(self.args.rundir, 'evolution_summary', 'mutation_fitness_changes_brain_overtime.pkl'), 'wb') as f:
                pickle.dump(fitness_changes_brain_overtime, f)
        else:
            fitness_changes_brain_overtime = {}
            fitness_changes_brain_overtime[gen] = np.array(ratio_offspring_parent_brain_mutation)
            with open(os.path.join(self.args.rundir, 'evolution_summary', 'mutation_fitness_changes_brain_overtime.pkl'), 'wb') as f:
                pickle.dump(fitness_changes_brain_overtime, f)
        # body mutations only, over time (every generation)
        if os.path.exists(os.path.join(self.args.rundir, 'evolution_summary', 'mutation_fitness_changes_body_overtime.pkl')):
            with open(os.path.join(self.args.rundir, 'evolution_summary', 'mutation_fitness_changes_body_overtime.pkl'), 'rb') as f:
                fitness_changes_body_overtime = pickle.load(f)
            fitness_changes_body_overtime[gen] = np.array(ratio_offspring_parent_body_mutation)
            with open(os.path.join(self.args.rundir, 'evolution_summary', 'mutation_fitness_changes_body_overtime.pkl'), 'wb') as f:
                pickle.dump(fitness_changes_body_overtime, f)
        else:
            fitness_changes_body_overtime = {}
            fitness_changes_body_overtime[gen] = np.array(ratio_offspring_parent_body_mutation)
            with open(os.path.join(self.args.rundir, 'evolution_summary', 'mutation_fitness_changes_body_overtime.pkl'), 'wb') as f:
                pickle.dump(fitness_changes_body_overtime, f)
        # brain mutations only, over time (every generation) - distilled
        if os.path.exists(os.path.join(self.args.rundir, 'evolution_summary', 'mutation_fitness_changes_brain_distilled_overtime.pkl')):
            with open(os.path.join(self.args.rundir, 'evolution_summary', 'mutation_fitness_changes_brain_distilled_overtime.pkl'), 'rb') as f:
                fitness_changes_brain_distilled_overtime = pickle.load(f)
            fitness_changes_brain_distilled_overtime[gen] = np.array(ratio_offspring_parent_brain_mutation_distilled)
            with open(os.path.join(self.args.rundir, 'evolution_summary', 'mutation_fitness_changes_brain_distilled_overtime.pkl'), 'wb') as f:
                pickle.dump(fitness_changes_brain_distilled_overtime, f)
        else:
            fitness_changes_brain_distilled_overtime = {}
            fitness_changes_brain_distilled_overtime[gen] = np.array(ratio_offspring_parent_brain_mutation_distilled)
            with open(os.path.join(self.args.rundir, 'evolution_summary', 'mutation_fitness_changes_brain_distilled_overtime.pkl'), 'wb') as f:
                pickle.dump(fitness_changes_brain_distilled_overtime, f)
        # body mutations only, over time (every generation) - distilled
        if os.path.exists(os.path.join(self.args.rundir, 'evolution_summary', 'mutation_fitness_changes_body_distilled_overtime.pkl')):
            with open(os.path.join(self.args.rundir, 'evolution_summary', 'mutation_fitness_changes_body_distilled_overtime.pkl'), 'rb') as f:
                fitness_changes_body_distilled_overtime = pickle.load(f)
            fitness_changes_body_distilled_overtime[gen] = np.array(ratio_offspring_parent_body_mutation_distilled)
            with open(os.path.join(self.args.rundir, 'evolution_summary', 'mutation_fitness_changes_body_distilled_overtime.pkl'), 'wb') as f:
                pickle.dump(fitness_changes_body_distilled_overtime, f)
        else:
            fitness_changes_body_distilled_overtime = {}
            fitness_changes_body_distilled_overtime[gen] = np.array(ratio_offspring_parent_body_mutation_distilled)
            with open(os.path.join(self.args.rundir, 'evolution_summary', 'mutation_fitness_changes_body_distilled_overtime.pkl'), 'wb') as f:
                pickle.dump(fitness_changes_body_distilled_overtime, f)
        # brain mutations only, over time (every generation) - regular
        if os.path.exists(os.path.join(self.args.rundir, 'evolution_summary', 'mutation_fitness_changes_brain_regular_overtime.pkl')):
            with open(os.path.join(self.args.rundir, 'evolution_summary', 'mutation_fitness_changes_brain_regular_overtime.pkl'), 'rb') as f:
                fitness_changes_brain_regular_overtime = pickle.load(f)
            fitness_changes_brain_regular_overtime[gen] = np.array(ratio_offspring_parent_brain_mutation_regular)
            with open(os.path.join(self.args.rundir, 'evolution_summary', 'mutation_fitness_changes_brain_regular_overtime.pkl'), 'wb') as f:
                pickle.dump(fitness_changes_brain_regular_overtime, f)
        else:
            fitness_changes_brain_regular_overtime = {}
            fitness_changes_brain_regular_overtime[gen] = np.array(ratio_offspring_parent_brain_mutation_regular)
            with open(os.path.join(self.args.rundir, 'evolution_summary', 'mutation_fitness_changes_brain_regular_overtime.pkl'), 'wb') as f:
                pickle.dump(fitness_changes_brain_regular_overtime, f)
        # body mutations only, over time (every generation) - regular
        if os.path.exists(os.path.join(self.args.rundir, 'evolution_summary', 'mutation_fitness_changes_body_regular_overtime.pkl')):
            with open(os.path.join(self.args.rundir, 'evolution_summary', 'mutation_fitness_changes_body_regular_overtime.pkl'), 'rb') as f:
                fitness_changes_body_regular_overtime = pickle.load(f)
            fitness_changes_body_regular_overtime[gen] = np.array(ratio_offspring_parent_body_mutation_regular)
            with open(os.path.join(self.args.rundir, 'evolution_summary', 'mutation_fitness_changes_body_regular_overtime.pkl'), 'wb') as f:
                pickle.dump(fitness_changes_body_regular_overtime, f)
        else:
            fitness_changes_body_regular_overtime = {}
            fitness_changes_body_regular_overtime[gen] = np.array(ratio_offspring_parent_body_mutation_regular)
            with open(os.path.join(self.args.rundir, 'evolution_summary', 'mutation_fitness_changes_body_regular_overtime.pkl'), 'wb') as f:
                pickle.dump(fitness_changes_body_regular_overtime, f)

        # record positive body mutations (overtime)
        body_mutations_info = []
        body_mutations_info_distilled = []

        # record counts of outcompetitions
        distilled_outcompetes_nondistilled = 0
        distilled_outcompetes_distilled = 0
        nondistilled_outcompetes_nondistilled = 0
        nondistilled_outcompetes_distilled = 0

        # record counts of migrations (with competition)
        migrations_with_competition = 0

        for ind in population:
            # continue if the individual is not valid
            if ind.fitness is None:
                raise ValueError("Individuals must have fitness values to be added to the archive.")
            # determine the bins
            bin = self.determine_bins(ind)
            ind.self_bin = bin
            if settings.VERBOSE:
                print()
                print("-----------------")
                print(f"Individual we try to add: {ind.self_id}")
                print(f" fitness: {ind.fitness}")
                print(f" bin: {bin}")
                if self.get_individual_with_id(ind.parent_id) is not None:
                    print(f" parent's fitness: {ind.parent_fitness}")
                    print(f" parent's bin: {ind.parent_bin}")

            #### record keeping
            info_dict = {}
            info_dict_distilled = {}
            if ind.self_id.startswith('BODY'):
                info_dict['self_id'] = ind.self_id
                info_dict['fitness'] = ind.fitness
                info_dict['self_bin'] = bin
                info_dict['parent_id'] = ind.parent_id
                if ind.parent_id == '':
                    info_dict['parent_fitness'] = None
                    info_dict['parent_bin'] = None
                else:
                    info_dict['parent_fitness'] = ind.parent_fitness
                    info_dict['parent_bin'] = ind.parent_bin
            if ind.self_id.startswith('BODY') and "DISTILLED" in ind.parent_id:
                info_dict_distilled['self_id'] = ind.self_id
                info_dict_distilled['fitness'] = ind.fitness
                info_dict_distilled['self_bin'] = bin
                info_dict_distilled['parent_id'] = ind.parent_id
                if ind.parent_id == '':
                    info_dict_distilled['parent_fitness'] = None
                    info_dict_distilled['parent_bin'] = None
                else:
                    info_dict_distilled['parent_fitness'] = ind.parent_fitness
                    info_dict_distilled['parent_bin'] = ind.parent_bin
            ####

            # update the archive
            if self.args.no_migration == False or (self.args.no_migration == True and gen < self.args.no_migration_start_gen):
                if self.archive[bin] is None:
                    #### record keeping
                    if ind.self_id.startswith('BODY'):
                        info_dict['positive_mutation'] = True
                        info_dict['with_competition'] = False
                    if ind.self_id.startswith('BODY') and "DISTILLED" in ind.parent_id:
                        info_dict_distilled['positive_mutation'] = True
                        info_dict_distilled['with_competition'] = False
                    # add the individual to the archive
                    self.archive[bin] = ind
                    if settings.VERBOSE:
                        print(f"Individual added to the archive because the bin is empty: {ind.self_id}")
                else:
                    current_fitness = self.archive[bin].fitness
                    if settings.VERBOSE:
                        print(f"Individual in the archive: {self.archive[bin].self_id}")
                        print(f" fitness: {self.archive[bin].fitness}")

                    if ind.fitness > current_fitness: # if the new individual has higher fitness, replace the old one

                        #### record keeping
                        if ind.self_id.startswith('BODY'):
                            info_dict['positive_mutation'] = True
                            info_dict['competition_fitness'] = current_fitness
                            info_dict['competition_id'] = self.archive[bin].self_id
                            info_dict['with_competition'] = True
                        if ind.self_id.startswith('BODY') and "DISTILLED" in ind.parent_id:
                            info_dict_distilled['positive_mutation'] = True
                            info_dict_distilled['competition_fitness'] = current_fitness
                            info_dict_distilled['competition_id'] = self.archive[bin].self_id
                            info_dict_distilled['with_competition'] = True
                        # determine whether this individual has migrated from a different bin
                        if ind.parent_bin == bin:
                            ... # no migration
                        else:
                            # migration with competition
                            if 'DISTILLED' in ind.self_id and 'DISTILLED' not in self.archive[bin].self_id:
                                distilled_outcompetes_nondistilled += 1
                            elif 'DISTILLED' in ind.self_id and 'DISTILLED' in self.archive[bin].self_id:
                                distilled_outcompetes_distilled += 1
                            elif 'DISTILLED' not in ind.self_id and 'DISTILLED' not in self.archive[bin].self_id:
                                nondistilled_outcompetes_nondistilled += 1
                            elif 'DISTILLED' not in ind.self_id and 'DISTILLED' in self.archive[bin].self_id:
                                nondistilled_outcompetes_distilled += 1
                            migrations_with_competition += 1

                        # add the individual to the archive
                        self.archive[bin] = ind
                        if settings.VERBOSE:
                            print(f"Individual added to the archive because it has higher fitness: {ind.self_id}")

                    else: # if the new individual has lower fitness, do not add it to the archive
                        if settings.VERBOSE:
                            print(f"Individual not added to the archive because it has lower fitness: {ind.self_id}")
                        #### record keeping
                        if ind.self_id.startswith('BODY'):
                            info_dict['positive_mutation'] = False
                            info_dict['competition_fitness'] = current_fitness
                            info_dict['competition_id'] = self.archive[bin].self_id
                            info_dict['with_competition'] = True
                        if ind.self_id.startswith('BODY') and "DISTILLED" in ind.parent_id:
                            info_dict_distilled['positive_mutation'] = False
                            info_dict_distilled['competition_fitness'] = current_fitness
                            info_dict_distilled['competition_id'] = self.archive[bin].self_id
                            info_dict_distilled['with_competition'] = True

            elif self.args.no_migration == True:
                if ind.parent_id == '':
                    if len(self) < 55:
                        # place the individual in to first empty bin
                        for key, value in self.archive.items():
                            if value is None:
                                self.archive[key] = ind
                                if settings.VERBOSE:
                                    print(f"Individual added to the archive because the bin is empty: {ind.self_id}")
                                break
                    ## this is a new individual
                    ## determine its bin
                    #bin = self.determine_bins(ind)
                    ## add the individual to the archive, if the bin is empty
                    #if self.archive[bin] is None:
                    #    self.archive[bin] = ind
                    #    if settings.VERBOSE:
                    #        print(f"Individual added to the archive because the bin is empty: {ind.self_id}")
                    #else:
                    #    if settings.VERBOSE:
                    #        print(f"Individual not added to the archive because the bin is not empty: {ind.self_id}")
                    continue
                # this will essentially be a parallel hillclimber, 
                # just compare the fitness of the new individual with its parent
                if ind.fitness > self.get_individual_with_id(ind.parent_id).fitness:
                    #### record keeping
                    if ind.self_id.startswith('BODY'):
                        info_dict['positive_mutation'] = True
                        info_dict['with_competition'] = True
                        info_dict['competition_fitness'] = ind.parent_fitness
                        info_dict['competition_id'] = ind.parent_id
                    if ind.self_id.startswith('BODY') and "DISTILLED" in ind.parent_id:
                        info_dict_distilled['positive_mutation'] = True
                        info_dict_distilled['with_competition'] = True
                        info_dict_distilled['competition_fitness'] = ind.parent_fitness
                        info_dict_distilled['competition_id'] = ind.parent_id

                    if settings.VERBOSE:
                        print(f"parallel hillclimber: Individual added to the archive because it has higher fitness: {ind.self_id}")
                        print(f"fitness: {ind.fitness}")
                        print(f"parent's fitness: {ind.parent_fitness}")
                    # find the parent's bin (not its bin value saved in the individual, but the actual bin in the archive)
                    for key, value in self.archive.items():
                        if value is None:
                            continue
                        if value.self_id == ind.parent_id:
                            parent_bin = key
                            break
                    # add the individual to the archive
                    self.archive[parent_bin] = ind
                else:
                    if settings.VERBOSE:
                        print(f"parallel hillclimber: Individual not added to the archive because it has lower fitness: {ind.self_id}")
                        print(f"fitness: {ind.fitness}")
                        print(f"parent's fitness: {ind.parent_fitness}")
                    #### record keeping
                    if ind.self_id.startswith('BODY'):
                        info_dict['positive_mutation'] = False
                        info_dict['with_competition'] = True
                        info_dict['competition_fitness'] = ind.parent_fitness
                        info_dict['competition_id'] = ind.parent_id
                    if ind.self_id.startswith('BODY') and "DISTILLED" in ind.parent_id:
                        info_dict_distilled['positive_mutation'] = False
                        info_dict_distilled['with_competition'] = True
                        info_dict_distilled['competition_fitness'] = ind.parent_fitness
                        info_dict_distilled['competition_id'] = ind.parent_id

            else:
                raise ValueError("Invalid no_migration setting")

            if ind.self_id.startswith('BODY'):
                body_mutations_info.append(info_dict)
            if ind.self_id.startswith('BODY') and "DISTILLED" in ind.parent_id:
                body_mutations_info_distilled.append(info_dict_distilled)

        # record positive body mutations
        if os.path.exists(os.path.join(self.args.rundir, 'evolution_summary', 'positive_body_mutations.pkl')):
            with open(os.path.join(self.args.rundir, 'evolution_summary', 'positive_body_mutations.pkl'), 'rb') as f:
                positive_body_mutations = pickle.load(f)
            positive_body_mutations[gen] = body_mutations_info
            with open(os.path.join(self.args.rundir, 'evolution_summary', 'positive_body_mutations.pkl'), 'wb') as f:
                pickle.dump(positive_body_mutations, f)
        else:
            positive_body_mutations = {}
            positive_body_mutations[gen] = body_mutations_info
            with open(os.path.join(self.args.rundir, 'evolution_summary', 'positive_body_mutations.pkl'), 'wb') as f:
                pickle.dump(positive_body_mutations, f)

        # record positive body mutations - distilled
        if os.path.exists(os.path.join(self.args.rundir, 'evolution_summary', 'positive_body_mutations_distilled.pkl')):
            with open(os.path.join(self.args.rundir, 'evolution_summary', 'positive_body_mutations_distilled.pkl'), 'rb') as f:
                positive_body_mutations_distilled = pickle.load(f)
            positive_body_mutations_distilled[gen] = body_mutations_info_distilled
            with open(os.path.join(self.args.rundir, 'evolution_summary', 'positive_body_mutations_distilled.pkl'), 'wb') as f:
                pickle.dump(positive_body_mutations_distilled, f)
        else:
            positive_body_mutations_distilled = {}
            positive_body_mutations_distilled[gen] = body_mutations_info_distilled
            with open(os.path.join(self.args.rundir, 'evolution_summary', 'positive_body_mutations_distilled.pkl'), 'wb') as f:
                pickle.dump(positive_body_mutations_distilled, f)

        # record counts of outcompetitions separately as npy arrays
        if os.path.exists(os.path.join(self.args.rundir, 'evolution_summary', 'distilled_outcompetes_nondistilled.npy')):
            distilled_outcompetes_nondistilled_arr = np.load(os.path.join(self.args.rundir, 'evolution_summary', 'distilled_outcompetes_nondistilled.npy'))
            distilled_outcompetes_nondistilled_arr = np.append(distilled_outcompetes_nondistilled_arr, distilled_outcompetes_nondistilled)
            np.save(os.path.join(self.args.rundir, 'evolution_summary', 'distilled_outcompetes_nondistilled.npy'), distilled_outcompetes_nondistilled_arr)
        else:
            np.save(os.path.join(self.args.rundir, 'evolution_summary', 'distilled_outcompetes_nondistilled.npy'), np.array([distilled_outcompetes_nondistilled]))
        if os.path.exists(os.path.join(self.args.rundir, 'evolution_summary', 'distilled_outcompetes_distilled.npy')):
            distilled_outcompetes_distilled_arr = np.load(os.path.join(self.args.rundir, 'evolution_summary', 'distilled_outcompetes_distilled.npy'))
            distilled_outcompetes_distilled_arr = np.append(distilled_outcompetes_distilled_arr, distilled_outcompetes_distilled)
            np.save(os.path.join(self.args.rundir, 'evolution_summary', 'distilled_outcompetes_distilled.npy'), distilled_outcompetes_distilled_arr)
        else:
            np.save(os.path.join(self.args.rundir, 'evolution_summary', 'distilled_outcompetes_distilled.npy'), np.array([distilled_outcompetes_distilled]))
        if os.path.exists(os.path.join(self.args.rundir, 'evolution_summary', 'nondistilled_outcompetes_nondistilled.npy')):
            nondistilled_outcompetes_nondistilled_arr = np.load(os.path.join(self.args.rundir, 'evolution_summary', 'nondistilled_outcompetes_nondistilled.npy'))
            nondistilled_outcompetes_nondistilled_arr = np.append(nondistilled_outcompetes_nondistilled_arr, nondistilled_outcompetes_nondistilled)
            np.save(os.path.join(self.args.rundir, 'evolution_summary', 'nondistilled_outcompetes_nondistilled.npy'), nondistilled_outcompetes_nondistilled_arr)
        else:
            np.save(os.path.join(self.args.rundir, 'evolution_summary', 'nondistilled_outcompetes_nondistilled.npy'), np.array([nondistilled_outcompetes_nondistilled]))
        if os.path.exists(os.path.join(self.args.rundir, 'evolution_summary', 'nondistilled_outcompetes_distilled.npy')):
            nondistilled_outcompetes_distilled_arr = np.load(os.path.join(self.args.rundir, 'evolution_summary', 'nondistilled_outcompetes_distilled.npy'))
            nondistilled_outcompetes_distilled_arr = np.append(nondistilled_outcompetes_distilled_arr, nondistilled_outcompetes_distilled)
            np.save(os.path.join(self.args.rundir, 'evolution_summary', 'nondistilled_outcompetes_distilled.npy'), nondistilled_outcompetes_distilled_arr)
        else:
            np.save(os.path.join(self.args.rundir, 'evolution_summary', 'nondistilled_outcompetes_distilled.npy'), np.array([nondistilled_outcompetes_distilled]))

        # record counts of migrations (with competition)
        if os.path.exists(os.path.join(self.args.rundir, 'evolution_summary', 'migrations_with_competition.npy')):
            migrations_with_competition_arr = np.load(os.path.join(self.args.rundir, 'evolution_summary', 'migrations_with_competition.npy'))
            migrations_with_competition_arr = np.append(migrations_with_competition_arr, migrations_with_competition)
            np.save(os.path.join(self.args.rundir, 'evolution_summary', 'migrations_with_competition.npy'), migrations_with_competition_arr)
        else:
            np.save(os.path.join(self.args.rundir, 'evolution_summary', 'migrations_with_competition.npy'), np.array([migrations_with_competition]))


        if settings.VERBOSE:
            print("-----------------")

    def determine_bins(self, ind):
        """Calculate the bin indices for the given individual."""
        # shape based bins
        active_voxels = ind.body.count_active_voxels();
        existing_voxels = ind.body.count_existing_voxels();
        if settings.VERBOSE:
            print(ind.self_id)
        return self.bin_from_counts(active_voxels, existing_voxels)

    def bin_from_counts(self, active_voxels, existing_voxels):
        nr_act_voxel_per_bin = np.round((self.args.bounding_box[0] * self.args.bounding_box[1]) / self.args.archive_bins[0])
        nr_exist_voxel_per_bin = np.round((self.args.bounding_box[0] * self.args.bounding_box[1]) / self.args.archive_bins[1])

        bin_active_voxels = (active_voxels // nr_act_voxel_per_bin); bin_active_voxels -= 1 if bin_active_voxels == self.args.archive_bins[0] else 0
        bin_existing_voxels = (existing_voxels // nr_exist_voxel_per_bin); bin_existing_voxels -= 1 if bin_existing_voxels == self.args.archive_bins[1] else 0
        bin_active_voxels = int(bin_active_voxels)
        bin_existing_voxels = int(bin_existing_voxels)

        if settings.VERBOSE:
            print(f"active voxels: {active_voxels}")
            print(f"existing voxels: {existing_voxels}")
            print(f"bin_active_voxels: {bin_active_voxels}")
            print(f"bin_existing_voxels: {bin_existing_voxels}")

        return (bin_active_voxels, bin_existing_voxels)

    def print_archive(self):
        """Print some useful information about the map."""
        # print the best fitness in the map
        print("Best fitness in the map: ", self.get_best_individual().fitness)
        # print the occupancy of the map
        print("Occupancy of the map: ", len(self), "/", len(self.archive))

    def get_fitnesses(self):
        """return a numpy array of fitnesses of the individuals in the map"""
        fitnesses = np.ones((self.args.archive_bins[0], self.args.archive_bins[1])) * -9999
        for i in range(0, self.args.archive_bins[0]):
            for j in range(0, self.args.archive_bins[1]):
                if self.archive[(i,j)] is not None:
                    fitnesses[i,j] = self.archive[(i,j)].fitness
        return fitnesses


if __name__ == '__main__':
    args = lambda: None
    args.archive_bins = (10, 10)
    args.bounding_box = (10, 10)
    archive = ARCHIVE(args)
    for i in range(101):
        print(archive.bin_from_counts(i, i))

        

