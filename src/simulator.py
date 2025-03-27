import multiprocessing
import gym
import numpy as np

from evogym.envs import *
from evogym import get_full_connectivity

from evogym_wrappers import ActionSkipWrapper, LocalObservationWrapper, GlobalObservationWrapper, LocalActionWrapper, GlobalActionWrapper, RewardShapingWrapper

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

def get_sim_pairs(population, **kwargs):
    sim_pairs = []
    for idx, ind in enumerate(population):
        sim_pairs.append( {'body':ind.body.to_phenotype(), 
                           'ind':ind,
                           'kwargs':kwargs} )
    return sim_pairs

def simulate_ind(sim_pair):
    # unpack the simulation pair
    body = sim_pair['body']
    ind = sim_pair['ind']
    kwargs = sim_pair['kwargs']
    # check if the individual has fitness already assigned (e.g. from previous subprocess run. sometimes process hangs and does not return, all the population is re-submitted to the queue)
    if ind.fitness is not None:
        return ind, ind.fitness
    # get the env
    env = make_env(body, **kwargs)
    # record keeping
    cum_rewards = []
    for i in range(5):
        cum_r = 0
        # run simulation
        obs = env.reset()
        #print(obs.shape) # (n_obs,) for standard, (nr_active_voxels, n_obs) for modular
        for t in range(500):
            # collect actions
            actions = ind.brain.get_action(obs)
            # step
            obs, r, d, i = env.step(actions)
            # record keeping
            cum_r += r
            # break if done
            if d:
                break
        cum_rewards.append(cum_r + 6.0) # push the reward to be positive
    return ind, np.mean(cum_rewards), np.std(cum_rewards), cum_rewards

def simulate_population(population, **kwargs):
    #get the simulator 
    sim_pairs = get_sim_pairs(population, **kwargs)
    # run the simulation
    finished = False
    while not finished:
        with multiprocessing.Pool(processes=len(sim_pairs)) as pool:
            results_f = pool.map_async(simulate_ind, sim_pairs)
            try:
                results = results_f.get(timeout=580)
                finished = True
            except multiprocessing.TimeoutError:
                print('TimeoutError')
                pass
    # assign fitness
    for r in results:
        ind, cum_reward_mean, cum_reward_std, cum_rewards = r
        for i in population:
            if i.self_id == ind.self_id:
                i.fitness = cum_reward_mean
                i.fitness_std = cum_reward_std
                i.fitnesses = cum_rewards




