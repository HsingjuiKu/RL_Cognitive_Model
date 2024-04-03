import gym
import paddle
import numpy as np
import random
import os
from parl.algorithms import DQN
from Agent import DQNAgent
from MLP import MLP
from ReplayBuffer import ReplayBuffer


def all_seed(env, seed=1):
    ''' omnipotent seed for RL, attention the position of seed function, you'd better put it just following the env create function
    Args:
        env (_type_):
        seed (int, optional): _description_. Defaults to 1.
    '''
    print(f"seed = {seed}")
    env.seed(seed)  # env config
    np.random.seed(seed)
    random.seed(seed)
    paddle.seed(seed)


def env_agent_config(cfg):
    ''' create env and agent
    '''
    env = gym.make(cfg['env_name'])
    if cfg['seed'] != 0:  # set random seed
        all_seed(env, seed=cfg["seed"])
    n_states = env.observation_space.shape[0]  # print(hasattr(env.observation_space, 'n'))
    n_actions = env.action_space.n  # action dimension
    print(f"n_states: {n_states}, n_actions: {n_actions}")
    cfg.update({"n_states": n_states, "n_actions": n_actions})  # update to cfg paramters
    model = MLP(n_states, n_actions)
    algo = DQN(model, gamma=cfg['gamma'], lr=cfg['lr'])
    memory = ReplayBuffer(cfg["memory_capacity"])  # replay buffer
    agent = DQNAgent(algo, memory, cfg)  # create agent
    return env, agent