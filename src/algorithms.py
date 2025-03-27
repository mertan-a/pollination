import os
import _pickle as pickle
import numpy as np
import time

from utils import get_files_in, create_folder_structure, natural_sort
from simulator import simulate_population
from make_gif import MAKEGIF
import settings
from pollination import pollinate

class MAP_ELITES():
    def __init__(self, args, archive):
        self.args = args
        self.archive = archive
        self.best_fitness = None

    def initialize_optimization(self):
        """
        initialize necessary things for MAP-Elites
        """
        # check if we are continuing or starting from scratch
        from_scratch = False
        if os.path.exists(os.path.join(self.args.rundir, 'pickled_population')):
            if settings.VERBOSE:
                print('found pickled population folder')
            pickles = get_files_in(os.path.join(
                self.args.rundir, 'pickled_population'))
            if len(pickles) == 0:
                from_scratch = True
                if settings.VERBOSE:
                    print('no pickles found')
        else:
            from_scratch = True
            if settings.VERBOSE:
                print('no pickled population folder found')

        if from_scratch:
            print('starting from scratch\n')
            create_folder_structure(self.args.rundir)
            self.starting_generation = 1
            self.current_generation = 0
        else:
            print('continuing from previous run\n')
            pickles = natural_sort(pickles, reverse=False)
            path_to_pickle = os.path.join(
                self.args.rundir, 'pickled_population', pickles[-1])
            self.archive = self.load_pickled_archive(path=path_to_pickle)
            # extract the generation number from the pickle file name
            self.starting_generation = int(
                pickles[-1].split('_')[-1].split('.')[0]) + 1
            if settings.VERBOSE:
                print(f'starting from generation {self.starting_generation}')

    def optimize(self):

        self.initialize_optimization()
        # write a file to indicate that the job is running
        with open(self.args.rundir + '/RUNNING', 'w') as f:
            if settings.VERBOSE:
                print('writing RUNNING file')
            pass

        for gen in range(self.starting_generation, self.args.nr_generations+1):

            # check if the job should be stopped due to time limit
            if self.args.slurm_queue and time.time() - settings.START_TIME > settings.MAX_TIME[self.args.slurm_queue]:
                print('time limit reached, stopping job')
                if settings.VERBOSE:
                    print(f"spent {time.time() - settings.START_TIME} seconds")
                settings.STOP = True
                break
            print('GENERATION: {}'.format(gen))
            self.do_one_generation(gen)
            self.record_keeping(gen)
            self.pickle_archive()

            if gen % self.args.gif_every == 0 or gen == self.args.nr_generations:
                t = MAKEGIF(self.args, self.archive.get_best_individual(), os.path.join(self.args.rundir, f'to_record/{gen}'))
                t.run()

            if self.args.pollination and gen % self.args.pollination_frequency == 0:
                pollinate(self.archive, self.args)

    def do_one_generation(self, gen):

        self.current_generation = gen
        print('PRODUCING OFFSPRINGS')
        offsprings = self.produce_offsprings()
        print('EVALUATING POPULATION')
        self.evaluate(offsprings)
        print('SELECTING NEW POPULATION')
        self.select(offsprings, gen)

    def produce_offsprings(self):
        '''produce offsprings from the current population
        '''
        offspring = self.archive.produce_offsprings(generation=self.current_generation)
        for i in range(self.args.nr_random_individual):
            offspring.append(self.archive.get_random_individual())
        if settings.VERBOSE:
            print()
            print('-----------------')
            print('offspring produced')
            print(f"number of offsprings: {len(offspring)}")
        return offspring

    def evaluate(self, population):
        '''evaluate the given population
        '''
        # evaluate the unevaluated individuals
        simulate_population(population=population, **vars(self.args))
        print('population evaluated\n')

    def select(self, population, gen):
        """ update the archive with the evaluated offsprings
        """
        self.archive.update_archive(population, gen)

    def pickle_archive(self):
        '''pickle the archive for later use
        '''
        pickle_dir = os.path.join(self.args.rundir, 'pickled_population')
        # find the current last generation pickle file
        f_names = get_files_in(pickle_dir)
        if len(f_names) > 0:
            f_names = natural_sort(f_names, reverse=False)
            last_file = f_names[-1]
            # load it, delete the brains and save it again
            last_pickle_file = os.path.join(pickle_dir, last_file)
            archive = self.load_pickled_archive(last_pickle_file)
            for ind in archive:
                ind.brain = None
            with open(last_pickle_file, 'wb') as f:
                pickle.dump(archive, f, protocol=-1)
        # save the current generation
        pickle_file = os.path.join(pickle_dir, 'generation_{}.pkl'.format(
            self.current_generation))
        with open(pickle_file, 'wb') as f:
            pickle.dump(self.archive, f, protocol=-1)

    def load_pickled_archive(self, path):
        '''load the population from a pickle file
        '''
        with open(path, 'rb') as f:
            archive = pickle.load(f)
        return archive

    def record_keeping(self, gen):
        '''writes a summary and saves the best individual'''

        #### FOT
        best_ind = self.archive.get_best_individual()
        # keep a fitness over time txt
        with open(os.path.join(self.args.rundir, 'evolution_summary', 'fitness_over_time.txt'), 'a') as f:
            f.write('{}\n'.format(best_ind.fitness))
        # keep a fitness over time npy
        if os.path.exists(os.path.join(self.args.rundir, 'evolution_summary', 'fitness_over_time.npy')):
            fitness_over_time = np.load(os.path.join(self.args.rundir, 'evolution_summary', 'fitness_over_time.npy'))
            fitness_over_time = np.append(fitness_over_time, best_ind.fitness)
            np.save(os.path.join(self.args.rundir, 'evolution_summary', 'fitness_over_time.npy'), fitness_over_time)
        else:
            np.save(os.path.join(self.args.rundir, 'evolution_summary', 'fitness_over_time.npy'), np.array([best_ind.fitness]))

        # keep a fitness std over time npy (mean of the stds of the individuals each generation)
        mean_std = np.mean([ind.fitness_std for ind in self.archive])
        if os.path.exists(os.path.join(self.args.rundir, 'evolution_summary', 'fitness_std_over_time.npy')):
            fitness_std_over_time = np.load(os.path.join(self.args.rundir, 'evolution_summary', 'fitness_std_over_time.npy'))
            fitness_std_over_time = np.append(fitness_std_over_time, mean_std)
            np.save(os.path.join(self.args.rundir, 'evolution_summary', 'fitness_std_over_time.npy'), fitness_std_over_time)
        else:
            np.save(os.path.join(self.args.rundir, 'evolution_summary', 'fitness_std_over_time.npy'), np.array([mean_std]))
        # keep a fitness std over time npy for non-distilled individuals
        mean_std_array = [ind.fitness_std for ind in self.archive if 'DISTILLED' not in ind.self_id]
        if len(mean_std_array) > 0:
            mean_std = np.mean(mean_std_array)
        else:
            mean_std = 0.0
        if os.path.exists(os.path.join(self.args.rundir, 'evolution_summary', 'fitness_std_over_time_non_distilled.npy')):
            fitness_std_over_time = np.load(os.path.join(self.args.rundir, 'evolution_summary', 'fitness_std_over_time_non_distilled.npy'))
            fitness_std_over_time = np.append(fitness_std_over_time, mean_std)
            np.save(os.path.join(self.args.rundir, 'evolution_summary', 'fitness_std_over_time_non_distilled.npy'), fitness_std_over_time)
        else:
            np.save(os.path.join(self.args.rundir, 'evolution_summary', 'fitness_std_over_time_non_distilled.npy'), np.array([mean_std]))
        # keep a fitness std over time npy for distilled individuals
        mean_std_array = [ind.fitness_std for ind in self.archive if 'DISTILLED' in ind.self_id]
        if len(mean_std_array) > 0:
            mean_std = np.mean(mean_std_array)
        else:
            mean_std = 0.0
        if os.path.exists(os.path.join(self.args.rundir, 'evolution_summary', 'fitness_std_over_time_distilled.npy')):
            fitness_std_over_time = np.load(os.path.join(self.args.rundir, 'evolution_summary', 'fitness_std_over_time_distilled.npy'))
            fitness_std_over_time = np.append(fitness_std_over_time, mean_std)
            np.save(os.path.join(self.args.rundir, 'evolution_summary', 'fitness_std_over_time_distilled.npy'), fitness_std_over_time)
        else:
            np.save(os.path.join(self.args.rundir, 'evolution_summary', 'fitness_std_over_time_distilled.npy'), np.array([mean_std]))

        #### DISTILLED IND COUNT
        # keep a count of the number of distilled individuals survived over time
        inds_ids = self.archive.get_individuals_ids()
        count = 0
        for i_id in inds_ids:
            if 'DISTILLED' in i_id:
                count += 1
        if os.path.exists(os.path.join(self.args.rundir, 'evolution_summary', 'distilled_individuals_count_over_time.npy')):
            count_over_time = np.load(os.path.join(self.args.rundir, 'evolution_summary', 'distilled_individuals_count_over_time.npy'))
            count_over_time = np.append(count_over_time, count)
            np.save(os.path.join(self.args.rundir, 'evolution_summary', 'distilled_individuals_count_over_time.npy'), count_over_time)
        else:
            np.save(os.path.join(self.args.rundir, 'evolution_summary', 'distilled_individuals_count_over_time.npy'), np.array([count]))

        #### DISTINCT DISTILLED INDS and HOW LONG THEY SURVIVED
        # keep track of the distinct distilled individuals and how long they survived
        # ids of distilled individuals in the current generation
        inds_ids = self.archive.get_individuals_ids()
        distilled_inds_ids = [i_id for i_id in inds_ids if 'DISTILLED' in i_id]
        # get the dictionary from disk
        if os.path.exists(os.path.join(self.args.rundir, 'evolution_summary', 'distilled_individuals_survival.pkl')):
            with open(os.path.join(self.args.rundir, 'evolution_summary', 'distilled_individuals_survival.pkl'), 'rb') as f:
                distilled_individuals_survival = pickle.load(f)
        else:
            distilled_individuals_survival = {}
        # update the dictionary
        for i_id in distilled_inds_ids:
            if i_id in distilled_individuals_survival:
                distilled_individuals_survival[i_id]['generations'].append(gen)
            else:
                # add the distilled individual to the dictionary
                distilled_individuals_survival[i_id] = {'generations': [gen]}
                # save individual's fitness
                ind = self.archive.get_individual_with_id(i_id)
                distilled_individuals_survival[i_id]['fitness'] = ind.fitness
                # save its bin
                distilled_individuals_survival[i_id]['bin'] = self.archive.determine_bins(ind)
        # save the dictionary back to disk
        with open(os.path.join(self.args.rundir, 'evolution_summary', 'distilled_individuals_survival.pkl'), 'wb') as f:
            pickle.dump(distilled_individuals_survival, f, protocol=-1)

        #### BEST INDIVIDUAL RELATED
        best_ind = self.archive.get_best_individual()
        # write the best individual
        with open(os.path.join(self.args.rundir, 'to_record', 'best.pkl'), 'wb') as f:
            pickle.dump(best_ind, f, protocol=-1)
        # check whether there is an improvement in the best fitness
        if self.best_fitness is None:
            self.best_fitness = best_ind.fitness
        else:
            if best_ind.fitness > self.best_fitness:
                self.best_fitness = best_ind.fitness

        #### PRINTING and SAVING ARCHIVE
        # also save the current archive 
        with open(os.path.join(self.args.rundir, 'to_record', 'archive.pkl'), 'wb') as f:
            pickle.dump(self.archive, f, protocol=-1)
        # print some useful stuf to the screen
        self.archive.print_archive()




