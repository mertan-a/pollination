import os
import _pickle as pickle
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams.update({'font.size': 22})
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from utils import natural_sort

def post_job_processing(args):
    # get the list of pickled populations
    pickle_dir = os.path.join(args.rundir, 'pickled_population')
    pickle_files = [f for f in os.listdir(pickle_dir) if f.endswith('.pkl')]
    # sort the pickle files
    pickle_files = natural_sort(pickle_files, reverse=False)

    # save each generation without brain for ease of analysis for body related stuff
    save_body_evolution_map_of_elites(pickle_dir, pickle_files)
    # save the fitness map of elites for each generation
    save_fitness_map(pickle_dir, pickle_files)
    # plot the map of elites for each generation
    plot_map_of_elites(pickle_dir, pickle_files)
    # measure the migration between bins for each generation
    measure_migration(pickle_dir, pickle_files)
    # plot migration over time plots
    plot_migration_over_time(pickle_dir, pickle_files)
    # plot the best individuals over time
    plot_best_fitness_over_time(pickle_dir, pickle_files)
    # delete pickled populations apart from the last one
    for pf in pickle_files[:-1]:
        os.remove(os.path.join(pickle_dir, pf))

def save_body_evolution_map_of_elites(pickle_dir, pickle_files):
    """ Save each generation's map of elites without brain for ease of analysis for body related stuff.

    Parameters
    ----------
    pickle_dir : str
        path to directory containing the pickle files
    pickle_files : list
        list of pickle files

    """
    print('saving body evolution map of elites')
    body_evolution = {}
    for pf in pickle_files:
        generation = int(pf.split('_')[-1].split('.')[0])
        print(generation)
        archive = load_pickled_population(os.path.join(pickle_dir, pf))
        for ind in archive:
            ind.brain = None
        body_evolution[generation] = archive
    # save it under evolution_summary folder
    saving_path = os.path.join( pickle_dir.replace('pickled_population', 'evolution_summary'), 'body_evolution.pkl')
    with open(saving_path, 'wb') as f:
        pickle.dump(body_evolution, f, protocol=-1)

def save_fitness_map(pickle_dir, pickle_files):
    """Save the fitness map for each generation as a single numpy array.

    Parameters
    ----------
    pickle_dir : str
        path to directory containing the pickle files
    pickle_files : list
        list of pickle files

    """
    print('saving fitness map')
    # read the body evolution map of elites file and do the job from there
    path = os.path.join(pickle_dir.replace('pickled_population', 'evolution_summary'), 'body_evolution.pkl')
    with open(path, 'rb') as f:
        body_evolution = pickle.load(f)
    # save each generation's fitness map as a numpy array
    fitness_maps = []
    for generation in body_evolution.keys():
        print(generation)
        map_of_elites = body_evolution[generation]
        # get the fitnesses for the non empty bins
        fitnesses = map_of_elites.get_fitnesses()
        fitness_maps.append(fitnesses)
    # save the list as a numpy array
    fitness_maps = np.array(fitness_maps)
    # save it under evolution_summary folder
    saving_path = os.path.join(pickle_dir.replace('pickled_population', 'evolution_summary'), 'fitness_maps.npy')
    np.save(saving_path, fitness_maps)

def plot_map_of_elites(pickle_dir, pickle_files):
    """ Plot the map of elites for each generation.
    Use white for empty niches, and a color gradient for the fitness of the individuals.

    Parameters
    ----------
    pickle_dir : str
        path to directory containing the pickle files
    pickle_files : list
        list of pickle files

    """
    print('plotting map of elites')
    # create a folder to save
    path = os.path.join(pickle_dir.replace('pickled_population', 'evolution_summary'), 'map_of_elites')
    if not os.path.exists(path):
        os.makedirs(path)
    # read the fitness maps file and do the job from there
    path = os.path.join(pickle_dir.replace('pickled_population', 'evolution_summary'), 'fitness_maps.npy')
    with open(path, 'rb') as f:
        fitness_maps = np.load(f)

    for i in range(fitness_maps.shape[0]):
        generation = i + 1
        print(generation)
        if generation % 100 != 0 and generation != 1 and generation != fitness_maps.shape[0]:
            continue
        # get the fitnesses and mask for the non empty bins
        fitnesses = fitness_maps[i]
        # plot the map of elites
        fig, ax = plt.subplots(figsize=(14, 14))
        # plot the map of elites
        masked_data = np.ma.masked_where(fitnesses == -9999, fitnesses)
        # shift the fitnesses to be positive
        masked_data += 5.0
        # plot
        im = plt.imshow(masked_data, cmap='jet', vmin=0, vmax=14)
        # invert the y axis
        plt.gca().invert_yaxis()
        # plot the colorbar
        cbar = plt.colorbar(im, orientation='vertical')
        # set the label for the colorbar
        cbar.set_label('Fitness')
        # set the labels for the axes
        plt.xlabel('Nr. of voxels')
        plt.ylabel('% of active voxels')
        # title
        plt.title(f'Generation {generation}')
        # save the figure
        plt.savefig(os.path.join(pickle_dir.replace('pickled_population', 'evolution_summary'), 'map_of_elites', f'generation_{generation}.png'))
        plt.close()
    # animate the map of elites
    # create a list of images
    images = []
    for i in range(fitness_maps.shape[0]):
        generation = i + 1
        if generation % 100 != 0 and generation != 1 and generation != fitness_maps.shape[0]:
            continue
        print(generation)
        # load the image
        img = plt.imread(os.path.join(pickle_dir.replace('pickled_population', 'evolution_summary'), 'map_of_elites', f'generation_{generation}.png'))
        images.append(img)
    pf = pickle_files[-1]
    generation = int(pf.split('_')[-1].split('.')[0])
    images.append(plt.imread(os.path.join(pickle_dir.replace('pickled_population', 'evolution_summary'), 'map_of_elites', f'generation_{generation}.png')))
    # save the animation with matplotlib
    fig = plt.figure(figsize=(14, 14))
    plt.axis('off')
    ims = [[plt.imshow(i, animated=True)] for i in images]
    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1000)
    ani.save(os.path.join(pickle_dir.replace('pickled_population', 'evolution_summary'), 'map_of_elites.gif'), writer='imagemagick')

def measure_migration(pickle_dir, pickle_files):
    """Count the number of migrations between bins at each generation.
    Also separately count the migrations that wins over a competition.
    Also keep track of where the migrations occur in the map spatially (one map for from and one for to per generation).
    Also keep track of the fitness values of the individuals that migrate as well as the best fitness value in the population at the time of migrations.

    Parameters
    ----------
    pickle_dir : str
        path to directory containing the pickle files
    pickle_files : list
        list of pickle files

    """
    print('measuring migration')
    # read the body evolution map of elites file and do the job from there
    path = os.path.join(pickle_dir.replace('pickled_population', 'evolution_summary'), 'body_evolution.pkl')
    with open(path, 'rb') as f:
        body_evolution = pickle.load(f)
    # keep track of where the migrations occur in the map spatially (one map for from and one for to per generation)
    migration_from = []
    migration_to = []
    # save it as a list, i.e. each element shows the number of migrations at that generation
    migration_numbers = []
    migration_with_competition_numbers = []
    # fitness values of the individuals that migrate (at each generation)
    fitness_values = []
    # keep a list of the ids of the individuals that have been checked
    checked_individuals = []
    # load each generation
    for generation in body_evolution.keys():
        print(generation)
        map_of_elites = body_evolution[generation]
        migration_from_map = np.zeros((map_of_elites.args.archive_bins[0], map_of_elites.args.archive_bins[1]))
        migration_to_map = np.zeros((map_of_elites.args.archive_bins[0], map_of_elites.args.archive_bins[1]))
        fitness_values.append([])
        # reset migration counters
        migration_counter = 0
        migration_with_competition_counter = 0
        # we will always be checking the previous generation
        if generation == 1:
            previous_map_of_elites = map_of_elites
            # save migration numbers
            migration_numbers.append(migration_counter)
            migration_with_competition_numbers.append(migration_with_competition_counter)
            # save the empty maps
            migration_from.append(migration_from_map)
            migration_to.append(migration_to_map)
            continue
        # check each individual
        for item in map_of_elites.archive.items():
            bin_id = item[0]
            ind = item[1]
            if ind is None:
                continue
            if ind.self_id in checked_individuals:
                continue
            if ind.parent_id == '':
                continue
            # we have an individual that has not been checked before and created through mutation
            parent_id = ind.parent_id
            # find its parent's bin from the previous generation
            for item in previous_map_of_elites.archive.items():
                parent_bin_id = item[0]
                parent_ind = item[1]
                if parent_ind is None:
                    continue
                if parent_ind.self_id == parent_id:
                    # we found the parent bin
                    break
            # check if the parent bin is the same as the current bin
            if parent_bin_id == bin_id:
                # no migration
                ...
            else:
                # migration
                migration_counter += 1
                # save the migration in the map
                migration_from_map[parent_bin_id[0]-1, parent_bin_id[1]-1] += 1
                migration_to_map[bin_id[0]-1, bin_id[1]-1] += 1
                # save the fitness value of the individual
                fitness_values[-1].append((ind.fitness, map_of_elites.get_best_fitness()))
                # check if the offspring's bin was empty in the previous generation
                if previous_map_of_elites.archive[bin_id] is None:
                    # no competition
                    ...
                else:
                    # competition
                    migration_with_competition_counter += 1
            # add the individual to the checked individuals
            checked_individuals.append(ind.self_id)
        # save the number of migrations for the generation
        migration_numbers.append(migration_counter)
        migration_with_competition_numbers.append(migration_with_competition_counter)
        # save the maps
        migration_from.append(migration_from_map)
        migration_to.append(migration_to_map)
        # set the current map of elites as the previous map of elites
        previous_map_of_elites = map_of_elites
    # save the list as a numpy array
    migration_numbers = np.array(migration_numbers)
    migration_with_competition_numbers = np.array(migration_with_competition_numbers)
    # save the maps as numpy arrays
    migration_from = np.array(migration_from)
    migration_to = np.array(migration_to)
    # save it under evolution_summary folder
    saving_path = os.path.join(pickle_dir.replace('pickled_population', 'evolution_summary'), 'migration_numbers.npy')
    np.save(saving_path, migration_numbers)
    saving_path = os.path.join(pickle_dir.replace('pickled_population', 'evolution_summary'), 'migration_with_competition_numbers.npy')
    np.save(saving_path, migration_with_competition_numbers)
    # save the maps
    saving_path = os.path.join(pickle_dir.replace('pickled_population', 'evolution_summary'), 'migration_from.npy')
    np.save(saving_path, migration_from)
    saving_path = os.path.join(pickle_dir.replace('pickled_population', 'evolution_summary'), 'migration_to.npy')
    np.save(saving_path, migration_to)
    # save the fitness values
    saving_path = os.path.join(pickle_dir.replace('pickled_population', 'evolution_summary'), 'fitness_values_migration.npy')
    with open(saving_path, 'wb') as f:
        pickle.dump(fitness_values, f, protocol=-1)

def plot_migration_over_time(pickle_dir, pickle_files):
    """Plot the migration over time.

    Parameters
    ----------
    pickle_dir : str
        path to directory containing the pickle files
    pickle_files : list
        list of pickle files

    """
    print('plotting migration over time')
    # load the migration numbers
    migration_numbers = np.load(os.path.join(pickle_dir.replace('pickled_population', 'evolution_summary'), 'migration_numbers.npy'))
    migration_with_competition_numbers = np.load(os.path.join(pickle_dir.replace('pickled_population', 'evolution_summary'), 'migration_with_competition_numbers.npy'))
    # plot the migration numbers
    fig, ax = plt.subplots()
    plt.plot(migration_numbers, label='Migration', linewidth=0.2)
    plt.plot(migration_with_competition_numbers, label='Migration with competition', linewidth=0.2)
    plt.xlabel('Generation')
    plt.ylabel('Number of migrations')
    plt.legend()
    plt.savefig(os.path.join(pickle_dir.replace('pickled_population', 'evolution_summary'), 'migration_over_time.png'))
    plt.close()
    # plot cumulative migration numbers
    fig, ax = plt.subplots()
    plt.plot(np.cumsum(migration_numbers), label='Migration')
    plt.plot(np.cumsum(migration_with_competition_numbers), label='Migration with competition')
    plt.xlabel('Generation')
    plt.ylabel('Cumulative number of migrations')
    plt.legend()
    plt.savefig(os.path.join(pickle_dir.replace('pickled_population', 'evolution_summary'), 'cumulative_migration_over_time.png'))
    plt.close()
    # plot the migration numbers over time, summed over the last 1000 generations
    fig, ax = plt.subplots()
    plt.plot(np.convolve(migration_numbers, np.ones(1000), mode='valid'), label='Migration')
    plt.xlabel('Generation')
    plt.ylabel('# of Migrations in\nthe last 1000 generations')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(pickle_dir.replace('pickled_population', 'evolution_summary'), 'sum_migration_over_time.png'))
    plt.close()

def plot_best_fitness_over_time(pickle_dir, pickle_files):
    '''Plot the best fitness over time for each generation.

    Parameters
    ----------
    pickle_dir : str
        path to directory containing the pickle files
    pickle_files : list
        list of pickle files

    '''
    print('plotting best fitness over time')
    # load the fitness over time txt
    with open(os.path.join(pickle_dir.replace('pickled_population', 'evolution_summary'), 'fitness_over_time.txt'), 'r') as f:
        best_fitnesses = f.readlines()
    best_fitnesses = [float(f.strip()) for f in best_fitnesses]
    # plot the best fitness over time
    fig, ax = plt.subplots()
    plt.plot(best_fitnesses)
    plt.xlabel('Generation')
    plt.ylabel('Best fitness')
    plt.savefig(os.path.join(pickle_dir.replace('pickled_population', 'evolution_summary'), 'best_fitness_over_time.png'))
    plt.close()

####################################

def save_best_individuals_over_time(pickle_dir, pickle_files):
    """Save the each different best individual over time.

    Parameters
    ----------
    pickle_dir : str
        path to directory containing the pickle files
    pickle_files : list
        list of pickle files

    """
    print('saving best individuals over time')
    # save it as a dictionary with generation number as key and the best individual as value
    best_individuals = {}
    for pf in pickle_files:
        generation = int(pf.split('_')[-1].split('.')[0])
        # load the population
        population = load_pickled_population(os.path.join(pickle_dir, pf))
        best_ind = population[0]
        # save if it is a new best individual
        # get the last saved best individual
        gens = list(best_individuals.keys())
        if len(gens) > 0:
            last_gen = gens[-1]
            last_best_ind = best_individuals[last_gen]
            if best_ind.self_id != last_best_ind.self_id:
                best_individuals[generation] = best_ind
                print(f'generation {generation} best individual {best_ind.self_id}')
        else:
            best_individuals[generation] = best_ind
            print(f'generation {generation} best individual {best_ind.self_id}')
    # save the dictionary
    with open(os.path.join(pickle_dir.replace('pickled_population', 'evolution_summary'), 'best_individuals_over_time.pkl'), 'wb') as f:
        pickle.dump(best_individuals, f, protocol=-1)

def save_mutation_numbers(pickle_dir, pickle_files):
    """Save the number of mutations for each generation.

    Parameters
    ----------
    pickle_dir : str
        path to directory containing the pickle files
    pickle_files : list
        list of pickle files

    """
    print('saving mutation numbers')
    # save it as a dictionary with generation number as key and number of mutations as another dictionary
    # with keys being the mutation types and values being the number of mutations of that type
    mutation_numbers = {}
    checked_individuals = []
    for pf in pickle_files:
        generation = int(pf.split('_')[-1].split('.')[0])
        # load the population
        population = load_pickled_population(os.path.join(pickle_dir, pf))
        # count the number of mutations for the whole population
        brain_mutations = 0
        body_mutations = 0
        for ind in population:
            if ind.self_id in checked_individuals:
                continue
            else:
                checked_individuals.append(ind.self_id)
            # add the mutation of the individual to the total number of mutations
            if "BODY" in ind.self_id:
                body_mutations += 1
            elif "BRAIN" in ind.self_id:
                brain_mutations += 1
            else:
                # it is a new individual
                ...
        # save the number of mutations for the generation
        mutation_numbers[generation] = {'brain': brain_mutations, 'body': body_mutations}
        print(f'generation {generation} brain mutations {brain_mutations} body mutations {body_mutations}')
    # save the dictionary under evolution_summary
    saving_path = os.path.join(pickle_dir.replace('pickled_population', 'evolution_summary'), 'mutation_numbers_over_time.pkl')
    with open(saving_path, 'wb') as f:
        pickle.dump(mutation_numbers, f, protocol=-1)

def save_summary_statistics(pickle_dir, pickle_files):
    """Save some summary statistics.

    Parameters
    ----------
    pickle_dir : str
        path to directory containing the pickle files
    pickle_files : list
        list of pickle files

    """
    # start a string to save the summary statistics
    summary_statistics = ''
    #### save best fitness
    last_population = load_pickled_population(os.path.join(pickle_dir, pickle_files[-1]))
    summary_statistics += f'(1) Best fitness: {last_population[0].fitness}\n'
    #### save the distances between the champion's body and its ancestor's body
    # load the champion's lineage
    with open(os.path.join(pickle_dir.replace('pickled_population', 'evolution_summary'), 'champions_lineage.pkl'), 'rb') as f:
        champions_lineage = pickle.load(f)
    if champions_lineage[-1][1].body.type == 'evolvable':
        # get the champion's body
        champion_body = champions_lineage[-1][1].body.structure
        # get the ancestor's body
        ancestor_body = champions_lineage[0][1].body.structure
        # calculate the distances
        shape_dist = edit_distance(champion_body, ancestor_body, shape_distances)
        active_passive_dist = edit_distance(champion_body, ancestor_body, active_passive_distances)
        material_dist = edit_distance(champion_body, ancestor_body, material_distances)
        summary_statistics += f'(2) Shape distance between the champion\'s body and its ancestor\'s body: {shape_dist}\n'
        summary_statistics += f'(3) Active/passive distance between the champion\'s body and its ancestor\'s body: {active_passive_dist}\n'
        summary_statistics += f'(4) Material distance between the champion\'s body and its ancestor\'s body: {material_dist}\n'
    #### ancestor vs champion generations
    summary_statistics += f'(5) Ancestor vs champion generations: {champions_lineage[0][0]} {champions_lineage[-1][0]}\n'
    #### count the number of mutations for the champion
    brain_mutations = 0
    body_mutations = 0
    for _, ind in champions_lineage:
        if "BODY" in ind.self_id:
            body_mutations += 1
        elif "BRAIN" in ind.self_id:
            brain_mutations += 1
        else:
            # it is a new individual
            ...
    summary_statistics += f'(6) Number of body brain total mutations for the champion: {body_mutations} {brain_mutations} {body_mutations + brain_mutations}\n'
    #### count the number of mutations for the whole population
    brain_mutations = 0
    body_mutations = 0
    with open(os.path.join(pickle_dir.replace('pickled_population', 'evolution_summary'), 'mutation_numbers_over_time.pkl'), 'rb') as f:
        mutation_numbers = pickle.load(f)
    for gen in mutation_numbers:
        brain_mutations += mutation_numbers[gen]['brain']
        body_mutations += mutation_numbers[gen]['body']
    summary_statistics += f'(7) Number of body brain total mutations for the whole population: {body_mutations} {brain_mutations} {body_mutations + brain_mutations}\n'
    #### TODO: check stagnation statistics
    #### 
    #### write the summary statistics to a file
    saving_path = os.path.join(pickle_dir.replace('pickled_population', 'evolution_summary'), 'summary_statistics.txt')
    with open(saving_path, 'w') as f:
        f.write(summary_statistics)

def save_champion_lineage(pickle_dir, pickle_files):
    """Save the lineage of the run's champion.

    Parameters
    ----------
    pickle_dir : str
        path to directory containing the pickle files
    pickle_files : list
        list of pickle files

    """
    print('saving champion lineage')
    # load the champion
    champion = load_pickled_population(os.path.join(pickle_dir, pickle_files[-1]))[0]
    generation = int(pickle_files[-1].split('_')[-1].split('.')[0])
    # save the champion's lineage
    champions_lineage = find_lineage(champion, generation, pickle_dir, pickle_files)
    for gen, ind in champions_lineage:
        print(gen, ind.self_id)
    # save it under evolution_summary folder
    saving_path = os.path.join( pickle_dir.replace('pickled_population', 'evolution_summary'), 'champions_lineage.pkl')
    with open(saving_path, 'wb') as f:
        pickle.dump(champions_lineage, f, protocol=-1)

def find_lineage(individual, generation, pickle_dir, pickle_files):
    """Find the lineage of an individual.

    Parameters
    ----------
    individual : Individual
        an individual
    generation : int
        generation number
    pickle_dir : str
        path to directory containing the pickle files
    pickle_files : list
        list of pickle files

    Returns
    -------
    list
        a list of tuples (generation, individual) representing the lineage of the individual

    """
    current_generation = generation
    current_individual_id = individual.self_id
    lineage = []
    while True:
        # finishing condition
        if current_individual_id == '':
            break
        elif current_generation == 0:
            pop = load_pickled_population(os.path.join(pickle_dir, pickle_files[current_generation]))
            for ind in pop:
                if ind.self_id == current_individual_id:
                    lineage.append( (current_generation, ind) )
            break
        # load the generation
        current_population = load_pickled_population(os.path.join(pickle_dir, pickle_files[current_generation]))
        current_population_individual_ids = [ind.self_id for ind in current_population]
        # check if the current individual is in the current population
        if current_individual_id in current_population_individual_ids:
            current_generation -= 1
            continue
        else:
            current_generation += 1
            pop = load_pickled_population(os.path.join(pickle_dir, pickle_files[current_generation]))
            for ind in pop:
                if ind.self_id == current_individual_id:
                    lineage.append( (current_generation, ind) )
                    current_individual_id = ind.parent_id
                    current_generation -= 1
                    break
    # reverse the lineage
    lineage.reverse()
    return lineage

def save_best_fitness_over_time(pickle_dir, pickle_files):
    """Save the best fitness over time for each generation as a numpy array.
    
    Parameters
    ----------
    pickle_dir : str
        path to directory containing the pickle files
    pickle_files : list
        list of pickle files

    """ 
    print('saving best fitness over time')
    best_fitness_over_time = []
    for pf in pickle_files:
        pop = load_pickled_population(os.path.join(pickle_dir, pf))
        assert max([ind.fitness for ind in pop]) == pop[0].fitness
        best_fitness_over_time.append(pop[0].fitness)
        print(pf, pop[0].fitness)
    # save it under evolution_summary folder
    saving_path = os.path.join( pickle_dir.replace('pickled_population', 'evolution_summary'), 'best_fitness_over_time.npy')
    np.save(saving_path, np.array(best_fitness_over_time))

def save_body_evolution(pickle_dir, pickle_files):
    """Save each generation without brain for ease of analysis for body related stuff.

    Parameters
    ----------
    pickle_dir : str
        path to directory containing the pickle files
    pickle_files : list
        list of pickle files

    """
    print('saving body evolution')
    body_evolution = {}
    for pf in pickle_files:
        generation = int(pf.split('_')[-1].split('.')[0])
        print(generation)
        pop = load_pickled_population(os.path.join(pickle_dir, pf))
        body_evolution[generation] = []
        for ind in pop:
            ind_information = {}
            ind_information['self_id'] = ind.self_id
            ind_information['parent_id'] = ind.parent_id
            ind_information['fitness'] = ind.fitness
            ind_information['detailed_fitness'] = ind.detailed_fitness
            ind_information['age'] = ind.age
            ind_information['body'] = ind.body
            body_evolution[generation].append(ind_information)
            print(ind_information)
    # save it under evolution_summary folder
    saving_path = os.path.join( pickle_dir.replace('pickled_population', 'evolution_summary'), 'body_evolution.pkl')
    with open(saving_path, 'wb') as f:
        pickle.dump(body_evolution, f, protocol=-1)

def combine_population_pkls(pickle_dir, pickle_files):
    """Combine all pickled populations into one pickle file.
    
    Parameters
    ----------
    pickle_dir : str
        path to directory containing the pickle files
    pickle_files : list
        list of pickle files
    
    """
    print('combining pickled populations')
    all_populations = {}
    for pf in pickle_files:
        generation = int(pf.split('_')[-1].split('.')[0])
        all_populations[generation] = load_pickled_population(os.path.join(pickle_dir, pf))
    with open(os.path.join(pickle_dir, 'all_populations.pkl'), 'wb') as f:
        pickle.dump(all_populations, f, protocol=-1)
    # remove all other pickle files
    for pf in pickle_files:
        os.remove(os.path.join(pickle_dir, pf))

def load_pickled_population(path):
    """Load the population from a pickle file.
    
    Parameters
    ----------
    path : str
        path to pickle file
    
    Returns
    -------
    object
        population object
    
    """
    with open(path, 'rb') as f:
        population = pickle.load(f)
    return population

    
