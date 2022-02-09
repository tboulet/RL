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

from MEMORY import Memory
from METRICS import *

from rl_algos_torch.DQN import DQN

def run(agent, env, episodes, wandb_cb = True, plt_cb = True, video_cb = True):
    print("Run starts.")
    
    config = agent.config
    if wandb_cb: 
        run = wandb.init(project="RL", 
                        entity="timotheboulet",
                        config=config
                        )
    if video_cb:
        env = Monitor(env, './div/videos', force=True, video_callable= lambda ep: ep % 100 == 0, )
    if plt_cb:
        logs = dict()
        
    k = 0
    for episode in range(episodes):
        k += 1
        done = False
        obs = env.reset()
        
        while not done:
            action = agent.act(obs)
            next_obs, reward, done, info = env.step(action)
            # print(obs, action, reward, done)
            metrics1 = agent.remember(obs, action, reward, done, next_obs, info)
            metrics2 = agent.learn()

            #Feedback
            for metric in metrics1 + metrics2:
                if wandb_cb:
                    wandb.log(metric, step = agent.step)
                    if video_cb:
                        #wandb.log({"gameplay": wandb.Video('./div/videos/')}, step = agent.step)
                        try:
                            if k % 10000 == 0: 
                                wandb.log({"gameplay": wandb.Video('./div/videos/', )}, step = agent.step)
                        except:
                            print('huh')
                        
                if plt_cb:
                    for key, value in metric.items():
                        try:
                            logs[key]["steps"].append(agent.step)
                            logs[key]["values"].append(value)    
                        except KeyError:
                            logs[key] = {"steps": [agent.step], "values": [value]}      
                        plt.clf()         
                        plt.plot(logs[key]["steps"][-100:], logs[key]["values"][-100:], '-b')
                        plt.title(key)
                        plt.savefig(f"figures/{key}")
                    
            #If episode ended.
            if done:
                break
            else:
                obs = next_obs
    
    if wandb_cb: run.finish()
    print("End of run.")




if __name__ == "__main__":
    #ENV
    env = gym.make("CartPole-v0")
    env = gym.make("LunarLander-v2")
    #MEMORY
    MEMORY_KEYS = ['observation', 'action','reward', 'done', 'next_observation']
    memory = Memory(MEMORY_KEYS=MEMORY_KEYS)
    #METRICS
    metrics = [Metric_Total_Reward, Metric_Epsilon, Metric_Actor_Loss, Metric_Critic_Loss, Metric_Critic_Value]
    #ACTOR PI
    actor = nn.Sequential(
        nn.Linear(in_features=env.observation_space.shape[0], out_features=32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, out_features=env.action_space.n),
        nn.Softmax(),
    )
    
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


    #AGENT
    agent = DQN(memory = memory, action_value=action_value, metrics = metrics)
    #agent = REINFORCE(memory=memory, actor=actor, metrics=metrics)
    
    #RUN
    run(agent, env, episodes=10000000000, wandb_cb = False, plt_cb=False, video_cb = False)









