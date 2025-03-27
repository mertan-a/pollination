import gym
import numpy as np
np.float = float
import imageio

from evogym.envs import *
from evogym import get_full_connectivity
from evogym_wrappers import RenderWrapper, ActionSkipWrapper, LocalObservationWrapper, GlobalObservationWrapper, LocalActionWrapper, GlobalActionWrapper, RewardShapingWrapper

class MAKEGIF():

    def __init__(self, args, ind, output_path):
        self.kwargs = vars(args)
        self.ind = ind
        self.output_path = output_path

    def run(self):
        body = self.ind.body.to_phenotype()
        connections = get_full_connectivity(body)
        env = gym.make('Walker-v0', body=body, connections=connections)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = RenderWrapper(env, render_mode='img')
        if 'sparse_acting' in self.kwargs and self.kwargs['sparse_acting']:
            env = ActionSkipWrapper(env, skip=self.kwargs['act_every'])
        if self.kwargs['controller'] in ['DECENTRALIZED']:
            env = LocalObservationWrapper(env, **self.kwargs)
            env = LocalActionWrapper(env, **self.kwargs)
        elif self.kwargs['controller'] in ['CENTRALIZED', 'CENTRALIZED_BIG']:
            env = GlobalObservationWrapper(env, **self.kwargs)
            env = GlobalActionWrapper(env, **self.kwargs)
        else:
            raise ValueError('Unknown controller', self.kwargs['controller'])
        env = RewardShapingWrapper(env)
        env.seed(17)
        env.action_space.seed(17)
        env.observation_space.seed(17)
        env.env.env.env.env.env.env._max_episode_steps = 500

        # run the environment
        cum_reward = 0
        observation = env.reset()
        for ts in range(500):
            action = self.ind.brain.get_action(observation)
            observation, reward, done, _ = env.step(action)
            cum_reward += reward
            if type(done) == bool:
                if done:
                    break
            elif type(done) == np.ndarray:
                if done.all():
                    break
            else:
                raise ValueError('Unknown type of done', type(d))
        cum_reward += 6.0 # push the reward to be positive
        imageio.mimsave(f"{self.output_path}_{cum_reward}.gif", env.imgs, duration=(1/50.0))
