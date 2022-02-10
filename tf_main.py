import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.initializers as ki
import tensorflow.keras.activations as ka
import tensorflow.keras.losses as losses
import tensorflow.keras.metrics as km
import tensorflow_probability as tfp

import sys
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import gym
import wandb

from div.render import render_agent

from MEMORY import Memory
from METRICS import *

from rl_algos.REINFORCE import REINFORCE
from rl_algos.DQN import DQN

def run(agent, env, episodes, wandb_cb = False, plt_cb = False):
    print("Run starts.")
    
    config = agent.config
    if wandb_cb: run = wandb.init(project="RL", 
                        entity="timotheboulet",
                        config=config
                        )
    if plt_cb:
        logs = dict()
        
    k = 0
    for episode in range(episodes):
        done = False
        obs = env.reset()
        

        while not done:
            action = agent.act(obs)
            next_obs, reward, done, info = env.step(action)
            metrics1 = agent.remember(obs, action, reward, done, next_obs, info)
            metrics2 = agent.learn()

            #Feedback
            k += 1
            for metric in metrics1 + metrics2:
                if wandb_cb:
                    wandb.log(metric, step = agent.step)
                if plt_cb:
                    for key, value in metric.items():
                        if key not in logs:
                            logs[key] = {"steps": [agent.step], "values": [value]}
                        else:
                            logs[key]["steps"].append(agent.step)
                            logs[key]["values"].append(value)          
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

    #MEMORY
    MEMORY_KEYS = ['observation', 'action','reward', 'done', 'next_observation']
    memory = Memory(MEMORY_KEYS=MEMORY_KEYS)

    #METRICS
    metrics = [Metric_Total_Reward, Metric_Epsilon, Metric_Actor_Loss, Metric_Critic_Loss, Metric_Critic_Value]

    #ACTOR PI
    actor = tf.keras.models.Sequential([
        kl.Dense(16, activation='tanh', kernel_initializer=ki.he_normal()),
        kl.Dense(16, activation='tanh', kernel_initializer=ki.he_normal()),
        kl.Dense(env.action_space.n, activation='softmax')
    ])
    #CRITIC Q/V
    action_value = tf.keras.models.Sequential([
        kl.Dense(16, activation='tanh', kernel_initializer=ki.he_normal()),
        kl.Dense(16, activation='tanh', kernel_initializer=ki.he_normal()),
        kl.Dense(env.action_space.n, activation='linear')
    ])



    #AGENT
    agent = DQN(memory = memory, action_value=action_value, metrics = metrics)
    #agent = REINFORCE(memory=memory, actor=actor, metrics=metrics)
    
    #RUN
    run(agent, env, episodes=1000, wandb_cb = True, plt_cb=False)









