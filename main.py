import tensorflow as tf
import tensorflow_probability as tfp
kl = tf.keras.layers
ki = tf.keras.initializers

import numpy as np
import gym
import sys
import math
import random
import matplotlib.pyplot as plt
import wandb

from div.render import render_agent

from MEMORY import Memory
from METRICS import *

from rl_algos.REINFORCE import REINFORCE
from rl_algos.DQN import DQN


def run(agent, env, episodes, wandb_cb = True):
    print("Run starts.")
    
    config = agent.config
    if wandb_cb: run = wandb.init(project="RL", 
                        entity="timotheboulet",
                        config=config
                        )   

    for episode in range(episodes):
        done = False
        obs = env.reset()

        while not done:
            action = agent.act(obs)
            next_obs, reward, done, info = env.step(action)
            metrics1 = agent.remember(obs, action, reward, done, next_obs, info)
            metrics2 = agent.learn()

            #Feedback
            for metric in metrics1 + metrics2:
                if wandb_cb:
                    wandb.log(metric, step = agent.step)
                    
            #If episode ended.
            if done:
                pass
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
    metrics = [Metric_Total_Reward, Metric_Epsilon, Metric_Actor_Loss]

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
    agent = REINFORCE(memory=memory, actor=actor, metrics=metrics)
    #RUN
    run(agent, env, episodes=1000, wandb_cb = True)









