import os
import itertools
import multiprocessing
import numpy as np
from copy import deepcopy
import gym
import _pickle as pickle
import torch
import settings
import random

from networks import NeuralNetwork, NeuralNetworkBig
from evogym import get_full_connectivity
from evogym_wrappers import ActionSkipWrapper, LocalObservationWrapper, GlobalObservationWrapper, LocalActionWrapper, GlobalActionWrapper, RewardShapingWrapper
from simulator import simulate_population
from utils import moore_neighbors, printv


def pollinate(archive, args):
    '''
    pollinates the archive by creating a distilled controller and replacing each individual's controller with the distilled one
    hyperparameters related to the distillation process are in the args object
    
    Parameters
    ----------
    archive : object
        the archive object that contains the population
    args : object
        the arguments object that contains the hyperparameters
    '''
    # gather dataset
    print('Gathering dataset')
    dataset, center_ind_id = gather_dataset(archive, args)
    # distill
    print('Distilling controller')
    distilled_controller = distill_controller(dataset, args)
    # replace
    print('Replacing controllers')
    replace_controller(archive, distilled_controller, args, center_ind_id)

def gather_dataset(archive, args):
    '''
    gathers dataset for distillation
    if replacement strategy not local
        runs the each individual in the archive many times and collects the (observation, action) pairs
    if replacement strategy local
        randomly chooses an individual from the archive, determines its neighbors, and collects the (observation, action) pairs of those
    to save time, we save the dataset to a file 
        (observation, action) pairs are saved per individual with their ids as the keys
    when gathering the dataset, we check if the dataset exists and load it if it does
    and we compare the ids of the individuals in the archive with the ids in the dataset to not run the same individual again

    returns the dataset
    
    Parameters
    ----------
    archive : object
        the archive object that contains the population
    args : object
        the arguments object that contains the hyperparameters

    Returns
    -------
    dataset : dict
        the dictionary that contains the (observation, action) pairs for each individual
        keys are the ids of the individuals, values are lists of (observation, action) pairs
    '''
    # check if the dataset exists
    if os.path.exists(os.path.join(args.rundir, 'dataset.pkl')):
        printv('old dataset exists, loading')
        with open(os.path.join(args.rundir, 'dataset.pkl'), 'rb') as f:
            dataset_in_disk = pickle.load(f)
    else:
        printv('old dataset does not exist')
        dataset_in_disk = {}
    # individuals we want in the dataset
    ids_in_dataset = []
    center_ind = None
    if args.pollination_replacement_strategy == 'LOCAL':
        # randomly choose an individual from the archive
        center_ind = random.choice(archive.get_individuals())
        # get the neighbors' bins
        neighbors_bins = moore_neighbors(center_ind.self_bin, args.pollination_local_radius, args.archive_bins)
        # get the ids of the individuals we want in the dataset
        for bin in neighbors_bins:
            if archive[bin] is not None:
                ids_in_dataset.append(archive[bin].self_id)
        ids_in_dataset.append(center_ind.self_id)
    else:
        ids_in_dataset = [ind.self_id for ind in archive]
    # make a list of individuals from the archive that are not in the dataset_in_disk
    individuals_to_simulate = [archive.get_individual_with_id(ind_id) for ind_id in ids_in_dataset if ind_id not in dataset_in_disk]
    print('number of individuals to simulate:', len(individuals_to_simulate))
    # make a list of individuals from the archive that are in the dataset_in_disk
    individual_ids_to_copy_from_dataset_in_disk = [ind_id for ind_id in ids_in_dataset if ind_id in dataset_in_disk]
    print('number of individuals to copy from old dataset:', len(individual_ids_to_copy_from_dataset_in_disk))
    # collect trajectories for the individuals that are not in the old dataset
    dataset_to_use = collect_trajectories(individuals_to_simulate, args)
    # copy the trajectories for the individuals that are in the old dataset
    for ind_id in individual_ids_to_copy_from_dataset_in_disk:
        dataset_to_use[ind_id] = dataset_in_disk[ind_id]
    # update the dataset in disk
    printv('updating the dataset in disk')
    for ind in individuals_to_simulate:
        dataset_in_disk[ind.self_id] = dataset_to_use[ind.self_id]
    with open(os.path.join(args.rundir, 'dataset.pkl'), 'wb') as f:
        pickle.dump(dataset_in_disk, f)
    if center_ind is not None:
        return dataset_to_use, center_ind.self_id
    else:
        return dataset_to_use, None

def distill_controller(dataset, args):
    ''' 
    trains a controller on the dataset
    saves some statistics about the training process
    returns the distilled controller

    Parameters
    ----------
    dataset : dict
        the dictionary that contains the (observation, action) pairs for each individual
        keys are the ids of the individuals, values are lists of (observation, action) pairs
    args : object
        the arguments object that contains the hyperparameters
    
    Returns
    -------
    distilled_controller : object
        the distilled controller
    '''
    # process the dataset
    # first, turn that into a list of (obs, act) pairs
    XY = []
    for item in dataset.values():
        XY.extend(item)
    if settings.VERBOSE:
        print('number of samples in the dataset:', len(XY))
    # shuffle the dataset
    np.random.shuffle(XY)
    # split the dataset into train and validation (80-20)
    XY_train = XY[:int(0.8*len(XY))]
    XY_val = XY[int(0.8*len(XY)):]
    if settings.VERBOSE:
        print('number of samples in the training dataset:', len(XY_train))
        print('number of samples in the validation dataset:', len(XY_val))
    # get the neural network
    if args.controller == 'CENTRALIZED':
        nn = NeuralNetwork(XY_train[0][0].shape[0], XY_train[0][1].shape[0])
    elif args.controller == 'CENTRALIZED_BIG':
        nn = NeuralNetworkBig(XY_train[0][0].shape[0], XY_train[0][1].shape[0])
    else:
        raise ValueError('Unknown controller', args.controller)
    nn.double()
    # optimizer
    optimizer = torch.optim.Adam(nn.parameters(), lr=1e-3)
    # train the network
    losses = []
    val_losses = []
    batch_size = 256
    number_of_steps = (len(XY_train) // batch_size) * args.num_distillation_epochs
    for i in range(number_of_steps):
        # get a random batch
        X = []
        Y = []
        # choose a random batch
        idx = np.random.randint(0, len(XY_train), batch_size)
        X = [XY_train[i][0] for i in idx]
        Y = [XY_train[i][1] for i in idx]
        # concatenate
        obs = np.stack(X, axis=0)
        act = np.stack(Y, axis=0)
        # convert to tensors
        obs = torch.from_numpy(obs).double()
        act = torch.from_numpy(act).double()
        # train
        est_act = nn(obs)
        # l1 loss
        loss = torch.mean(torch.abs(est_act - act))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # record keeping
        losses.append(loss.item())
        # validate
        if (i+1) % (number_of_steps//20) == 0:
            cum_val_loss = 0
            for i in range(0, len(XY_val), batch_size):
                # validate
                X = []
                Y = []
                # get the batch
                for j in range(i, min(i+batch_size, len(XY_val))):
                    X.append(XY_val[j][0])
                    Y.append(XY_val[j][1])
                # stack
                obs = np.stack(X, axis=0)
                act = np.stack(Y, axis=0)
                # convert to tensors
                obs = torch.from_numpy(obs).double()
                act = torch.from_numpy(act).double()
                # predict
                with torch.no_grad():
                    nn.eval()
                    est_act = nn(obs)
                    # l1 loss
                    loss = torch.mean(torch.abs(est_act - act)).item()
                    nn.train()
                cum_val_loss += loss
            val_loss = cum_val_loss / (len(XY_val) // batch_size)
            val_losses.append(val_loss)
            if settings.VERBOSE:
                print('Loss:', np.mean(losses[-number_of_steps//20:]), 'Val loss:', val_loss)
    if settings.VERBOSE:
        print('Distillation done')
    return nn

def replace_controller(archive, distilled_controller, args, center_ind_id):
    '''
    replaces the controller of *some* individuals in the archive with the distilled controller
    check the args object for the hyperparameters related to the replacement process

    Parameters
    ----------
    archive : object
        the archive object that contains the population
    distilled_controller : object
        the distilled controller
    args : object
        the arguments object that contains the hyperparameters
    center_ind_id : str
        only used if the replacement strategy is local, the id of the individual whose neighbors will be replaced
    '''

    individuals_to_pollinate = [] # individuals whose controllers will be replaced by the distilled controller (potentially -- maybe not all of them will be replaced)

    if args.pollination_replacement_strategy == 'BEST':

        # get a list of individuals sorted by fitness
        sorted_individuals = sorted(archive, key=lambda x: x.fitness, reverse=True)
        # determine the number of individuals to replace
        number_of_individuals_to_replace = int(args.pollination_ratio * len(archive))
        # collect the individuals to pollinate
        individuals_to_pollinate.extend(sorted_individuals[:number_of_individuals_to_replace])

        ########################################
        if settings.VERBOSE:
            print('replacing the best individuals')
            print('number of individuals to replace:', number_of_individuals_to_replace)
        ########################################

    elif args.pollination_replacement_strategy == 'WORST':

        # get a list of individuals sorted by fitness
        sorted_individuals = sorted(archive, key=lambda x: x.fitness, reverse=False)
        # determine the number of individuals to replace
        number_of_individuals_to_replace = int(args.pollination_ratio * len(archive))
        # collect the individuals to pollinate
        individuals_to_pollinate.extend(sorted_individuals[:number_of_individuals_to_replace])

        ########################################
        if settings.VERBOSE:
            print('replacing the worst individuals')
            print('number of individuals to replace:', number_of_individuals_to_replace)
        ########################################

    elif args.pollination_replacement_strategy == 'RANDOM':

        # list of individuals
        individuals = [ind for ind in archive if ind is not None]
        # determine the number of individuals to replace
        number_of_individuals_to_replace = int(args.pollination_ratio * len(archive))
        # collect the individuals to pollinate
        number_of_individuals_to_replace = min(number_of_individuals_to_replace, len(individuals)) # to avoid value error
        individuals_to_pollinate.extend(np.random.choice(individuals, number_of_individuals_to_replace, replace=False))

        ########################################
        if settings.VERBOSE:
            print('replacing random individuals')
            print('number of individuals to replace:', number_of_individuals_to_replace)
        ########################################

    elif args.pollination_replacement_strategy == 'LOCAL':

        # get the center individual
        center_ind = archive.get_individual_with_id(center_ind_id)
        # get the neighbors' bins
        neighbors_bins = moore_neighbors(center_ind.self_bin, args.pollination_local_radius, args.archive_bins)
        # collect the individuals to pollinate
        for bin in neighbors_bins:
            if archive[bin] is not None:
                individuals_to_pollinate.append(archive[bin])
        individuals_to_pollinate.append(center_ind)

        ########################################
        if settings.VERBOSE:
            print('replacing local individuals')
            print('number of individuals to replace:', len(individuals_to_pollinate))
        ########################################

    else:
        raise ValueError('Unknown replacement strategy', args.pollination_replacement_strategy)

    record = {} # keys: ids of individuals, values: dict with keys 'old_fitness', 'new_fitness', 'old_controller_model'
    for ind in individuals_to_pollinate:
        record[ind.self_id] = {}
        record[ind.self_id]['old_fitness'] = ind.fitness
        record[ind.self_id]['old_controller_model'] = deepcopy(ind.brain.model)
        ind.brain.set_model(deepcopy(distilled_controller))
        ind.fitness = None
    # re-evaluate the individuals
    for inds in itertools.zip_longest(*(iter(individuals_to_pollinate),) * args.nr_parents):
        # remove None
        inds = [ind for ind in inds if ind is not None]
        # run the simulation
        simulate_population(inds, **vars(args))
    # record the new fitness values and decide whether to keep the new controller
    replace_counter = 0
    for ind in individuals_to_pollinate:
        record[ind.self_id]['new_fitness'] = ind.fitness
        if args.replace_only_if_similar == True and record[ind.self_id]['new_fitness'] < record[ind.self_id]['old_fitness'] * 0.9: # don't replace case
            # if we only want to replace with a better controller and if the new controller is not better or similar, revert back to the old controller
            ind.brain.set_model(deepcopy(record[ind.self_id]['old_controller_model']))
            ind.fitness = record[ind.self_id]['old_fitness']

            ########################################
            if settings.VERBOSE:
                print('reverting back to the old controller')
                print('individual id:', ind.self_id)
                print('old fitness:', record[ind.self_id]['old_fitness'])
                print('new fitness:', record[ind.self_id]['new_fitness'])
            ########################################

        else: # replace case
            replace_counter += 1
            # if we want to replace with any controller or if the new controller is better or similar, keep the new controller
            ind.brain.set_model(deepcopy(distilled_controller))
            # decide which fitness value to keep
            if args.re_eval_pollinated == False:
                # keep the new fitness value
                ind.fitness = record[ind.self_id]['new_fitness']
            else:
                # keep the old fitness value
                ind.fitness = record[ind.self_id]['old_fitness']
            # rename the individual to record that its controller has been replaced
            id_tokens = ind.self_id.split('_')
            if len(id_tokens) == 2:
                ind.self_id = 'DISTILLED_' + ind.self_id
            elif len(id_tokens) == 3:
                ind.self_id = id_tokens[0] + '_' + 'DISTILLED_' + id_tokens[1] + '_' + id_tokens[2]
            elif len(id_tokens) == 4:
                ind.self_id = id_tokens[0] + '_' + id_tokens[1] + '+_' + id_tokens[2] + '_' + id_tokens[3]
            else:
                raise ValueError('Unknown individual id format', ind.self_id)

            ########################################
            if settings.VERBOSE:
                print('keeping the new controller')
                print('individual id:', ind.self_id, 'ind fitness:', ind.fitness)
                print('old fitness:', record['_'.join(id_tokens)]['old_fitness'])
                print('new fitness:', record['_'.join(id_tokens)]['new_fitness'])
            ########################################

        # record keeping
        if os.path.exists(os.path.join(args.rundir, 'evolution_summary', 'inserted_distilled_individuals_count_overtime.pkl')):
            with open(os.path.join(args.rundir, 'evolution_summary', 'inserted_distilled_individuals_count_overtime.pkl'), 'rb') as f:
                inserted_distilled_individuals_count_overtime = pickle.load(f)
        else:
            inserted_distilled_individuals_count_overtime = []
        inserted_distilled_individuals_count_overtime.append( {'inserted_count':replace_counter, 'considered_count':len(individuals_to_pollinate)} )
        with open(os.path.join(args.rundir, 'evolution_summary', 'inserted_distilled_individuals_count_overtime.pkl'), 'wb') as f:
            pickle.dump(inserted_distilled_individuals_count_overtime, f)

def collect_trajectories(individuals, args):
    '''
    collects trajectories for the given individuals
    runs the each individual many times and collects the (observation, action) pairs
    returns the dataset
    
    Parameters
    ----------
    individuals : list
        the list of individuals to collect the trajectories for
    args : object
        the arguments object that contains the hyperparameters

    Returns
    -------
    dataset : dict
        the dictionary that contains the (observation, action) pairs for each individual
        keys are the ids of the individuals, values are lists of (observation, action) pairs
    '''
    if settings.VERBOSE:
        print('collecting trajectories')
    # dataset
    dataset = {}
    # get this many individuals from the list
    number_of_simultaneous_simulations = args.nr_parents
    for inds in itertools.zip_longest(*(iter(individuals),) * number_of_simultaneous_simulations):
        to_send = [(ind, args) for ind in inds if ind is not None]
        if settings.VERBOSE:
            print('number of individuals to simulate:', len(to_send))
            print('individual ids:', [ind.self_id for ind in inds if ind is not None])
        # run the simulation
        finished = False
        while not finished:
            with multiprocessing.Pool(processes=len(to_send)) as pool:
                results_f = pool.map_async(simulate_ind, to_send)
                try:
                    results = results_f.get(timeout=580)
                    finished = True
                except multiprocessing.TimeoutError:
                    print('TimeoutError')
                    pass
        # save the results
        for ind, obs_action_pairs in results:
            dataset[ind.self_id] = obs_action_pairs
    return dataset

def simulate_ind(item):
    # unpack the simulation pair
    ind = item[0]
    body = ind.body.to_phenotype()
    args = item[1]
    # get the env
    env = make_env(body, **vars(args))
    # record trajectories
    obs_action_pairs = []
    for i in range(20):
        # run simulation
        obs = env.reset()
        #print(obs.shape) # (n_obs,) for standard, (nr_active_voxels, n_obs) for modular
        for t in range(500):
            # collect actions
            actions = ind.brain.get_action(obs)
            # record
            obs_action_pairs.append((obs, actions))
            # step
            obs, r, d, i = env.step(actions)
            # break if done
            if d:
                break
    return ind, obs_action_pairs

def make_env(body, **kwargs):
    np.float = float
    env = gym.make(kwargs['task'], body=body, connections=get_full_connectivity(body))
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if 'sparse_acting' in kwargs and kwargs['sparse_acting']:
        env = ActionSkipWrapper(env, skip=kwargs['act_every'])
    if kwargs['controller'] in ['DECENTRALIZED']:
        env = LocalObservationWrapper(env, **kwargs)
        env = LocalActionWrapper(env, **kwargs)
    elif kwargs['controller'] in ['CENTRALIZED', 'CENTRALIZED_BIG']:
        env = GlobalObservationWrapper(env, **kwargs)
        env = GlobalActionWrapper(env, **kwargs)
    else:
        raise ValueError('Unknown controller', kwargs['controller'])
    env = RewardShapingWrapper(env)
    env.seed(17)
    env.action_space.seed(17)
    env.observation_space.seed(17)
    env.env.env.env.env.env.env._max_episode_steps = 500
    return env
