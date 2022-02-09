import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import sys
import math
import random
import matplotlib.pyplot as plt
import numpy as np

import gym
from gym.wrappers import Monitor
import wandb

from div.render import render_agent
from div.run import run 
from div.utils import *

from MEMORY import Memory
from METRICS import *

from rl_algos_torch.DQN import DQN


class ObservationMarioWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def observation(self, obs):
        obs = np.swapaxes(obs, 0, 2)
        obs = np.array(obs)
        return obs


if __name__ == "__main__":
    #ENV
    env = load_smb_env(obs_complexity=1, action_complexity=2)
    env = ObservationMarioWrapper(env)
    #MEMORY
    MEMORY_KEYS = ['observation', 'action','reward', 'done', 'next_observation']
    memory = Memory(MEMORY_KEYS=MEMORY_KEYS)
    #METRICS
    metrics = [Metric_Total_Reward, Metric_Epsilon, Metric_Actor_Loss, Metric_Critic_Loss, Metric_Critic_Value]
    
    #CRITIC Q/V
    action_value = nn.Sequential(
        nn.Linear(in_features=env.observation_space.shape[0], out_features=32),
        nn.Tanh(),
        nn.Linear(32, 32),
        nn.Tanh(),
        nn.Linear(32, 32),
        nn.Tanh(),
        nn.Linear(32, out_features=env.action_space.n),
    )
    n_actions = env.action_space.n
    height, width, n_channels = env.observation_space.shape
    
    action_value = nn.Sequential(
        nn.Conv2d(n_channels, 8, 3),
        nn.Tanh(),
        nn.Conv2d(8, 8, 3),
        nn.Flatten(),
        nn.Linear(475776, n_actions),
    )

    #AGENT
    agent = DQN(memory = memory, action_value=action_value, metrics = metrics)
    
    #RUN
    render_agent(agent, env, show_metrics=True, episodes=2)
    run(agent, env, episodes=1000, wandb_cb = True, plt_cb=False, video_cb = False)
    







