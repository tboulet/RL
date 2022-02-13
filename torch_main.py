import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import sys
import math
import random
import os
from moviepy.editor import *
import matplotlib.pyplot as plt
import numpy as np

import gym
import wandb

from div.render import render_agent
from div.run import run 
from div.utils import *

from MEMORY import Memory
from METRICS import *

from rl_algos_torch._algos import DQN, REINFORCE, ACTOR_CRITIC
from CONFIGS import DQN_CONFIG, REINFORCE_CONFIG, ACTOR_CRITIC_CONFIG


if __name__ == "__main__":
    #ENV
    env = gym.make("CartPole-v0")
    # env = gym.make("LunarLander-v2")
    env = gym.make("Acrobot-v1")
    # env = gym.make("Breakout-v0")
    
    #MEMORY
    MEMORY_KEYS = ['observation', 'action','reward', 'done', 'next_observation']
    memory = Memory(MEMORY_KEYS=MEMORY_KEYS)
    #METRICS
    metrics = [
        MetricS_On_Learn,
        Metric_Total_Reward, 
        Metric_Epsilon, 
        Metric_Time_Count]
    
    #ACTOR PI
    actor = nn.Sequential(
        nn.Linear(in_features=env.observation_space.shape[0], out_features=32),
        nn.Tanh(),
        nn.Linear(32, out_features=env.action_space.n),
        nn.Softmax(),
    )
    
    #ACTION VALUE Q
    action_value = nn.Sequential(
        nn.Linear(in_features=env.observation_space.shape[0], out_features=18),
        nn.ReLU(),
        nn.Linear(18, out_features=env.action_space.n),
    )
    
    #STATE VALUE V
    state_value = nn.Sequential(
        nn.Linear(in_features=env.observation_space.shape[0], out_features=18),
        nn.ReLU(),
        nn.Linear(18, out_features=1),
    )

    #AGENT
    # agent = DQN(memory = memory, action_value=action_value, metrics = metrics)
    # agent = REINFORCE(memory=memory, actor=actor, metrics=metrics)
    agent = ACTOR_CRITIC(memory = memory, 
                         actor = actor, 
                         action_value=action_value, 
                         state_value=state_value,
                         metrics = metrics, 
                         config = ACTOR_CRITIC_CONFIG)
    
    
    #RUN
    run(agent, env, episodes=10000, wandb_cb = False, plt_cb=False, video_cb = False)
    render_agent(agent, env)







