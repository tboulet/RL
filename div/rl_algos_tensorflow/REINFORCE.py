import tensorflow as tf
import tensorflow_probability as tfp

from RL.CONFIGS import DUMMY_CONFIG
kl = tf.keras.layers
ki = tf.keras.initializers

import numpy as np
import gym

import sys
import math
import random
import matplotlib.pyplot as plt
import wandb

from RL.MEMORY import Memory
from RL.CONFIGS import REINFORCE_CONFIG

class REINFORCE():

    def __init__(self, memory, actor, metrics = [], config = REINFORCE_CONFIG):
        self.metrics = list(Metric(self) for Metric in metrics)
        self.step = 0
        self.memory = memory
        self.config = config

        self.actor = actor
        
        self.opt = tf.keras.optimizers.Adam(1e-4)
        self.gamma = 0.99

    def act(self, observation):
        observations = tf.expand_dims(observation, axis=0)  #(1, n_obs)
        probs = self.actor(observations)                    #(1, n_actions)
        probs = tfp.distributions.Categorical(probs=probs)  #categorical(1,n_actions)
        action = probs.sample()[0].numpy()                  
        return action

    def learn(self):
        metrics = list()
        self.step += 1

        observations, actions, rewards, dones, next_observations = self.memory.sample(      #(T, ?)
            method='all')
        if dones[-1]:
            ep_length = rewards.shape[0]                            #T
            actions = tf.expand_dims(                               #(T,)
                tf.constant(actions), axis=-1)
            action_inds = tf.expand_dims(tf.range(0, ep_length), axis=-1)   #(T, 1) , [[0], [1], ... [T-1]]
            action_inds = tf.concat((action_inds, actions), axis=-1)        #(T, 2)
            G = [rewards[-1].numpy()]
            for i in range(1, ep_length):
                prev_G = self.gamma * rewards[-i-1].numpy() + G[-i]
                G.insert(0, prev_G)
            G = tf.constant(G, dtype=tf.float32)                    #(T,)

            with tf.GradientTape() as tape:
                probs = self.actor(observations, training=True)     # [[PI_θ(a | s) for a in A] for t in range(T)] of shape (T, n_actions) 
                log_probs = tf.math.log(probs)                      #   log(PI_θ(a | s))
                log_probs = tf.gather_nd(log_probs, action_inds)    # (T, n_actions) g_nd (T, 2) = (T,)
                loss = - tf.math.multiply(G, log_probs)             # [log(PI_θ(a|s)) * R_future(s) for t in episode]

            grads = tape.gradient(loss, self.actor.trainable_weights)
            self.opt.apply_gradients(zip(grads, self.actor.trainable_weights))

            self.memory.__empty__()

            metrics = list(metric.on_learn(actor_loss = loss[0].numpy()) for metric in self.metrics)

        return metrics


    def remember(self, observation, action, reward, done, next_observation, info={}, **kwargs):
        self.memory.remember((observation, action, reward,
                            done, next_observation))
        return list(metric.on_remember(obs = observation, action = action, reward = reward, done = done, next_obs = next_observation) for metric in self.metrics)








if __name__ == "__main__":

    env = gym.make("CartPole-v0")

    actor = tf.keras.models.Sequential([
        kl.Dense(16, activation='tanh', kernel_initializer=ki.he_normal()),
        kl.Dense(16, activation='tanh', kernel_initializer=ki.he_normal()),
        kl.Dense(env.action_space.n, activation='softmax')
    ])

    MEMORY_KEYS = ['observation', 'action',
                       'reward', 'done', 'next_observation']
    memory = Memory(MEMORY_KEYS=MEMORY_KEYS, max_memory_len=40960)

    agent = REINFORCE(memory=memory, actor = actor)  
    
    
    
    
    #sys.exit()



    episodes = 10000
    L_rewards_tot = list()
    L_loss = list()
    moy = lambda L : sum(L) / len(L)
    reward_tot = 0
    plt.figure()
    plt.ion()

    obs = env.reset()
    for episode in range(episodes):
        done = False
        reward_tot = 0

        while not done:
            action = agent.act(obs)
            next_obs, reward, done, info = env.step(action)
            agent.remember(obs, action, reward, done, next_obs, info)
            metrics = agent.learn()
            if done:
                obs = env.reset()
            else:
                obs = next_obs

            try:
                L_loss.append(math.log(metrics["loss"]))
            except: 
                pass
            reward_tot += reward

        L_rewards_tot.append(reward_tot)
        plt.clf()
        plt.plot(L_rewards_tot[-100:], label = "total reward")
        plt.plot(L_loss[-100:], label = "actor loss (log)")
        plt.legend()
        plt.show()
        plt.pause(1e-3)
        

