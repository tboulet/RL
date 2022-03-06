import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import math
import gym
import sys
import random
import matplotlib.pyplot as plt

from tensorflow.keras import layers as kl
from tensorflow.keras import initializers as ki
from tensorflow.keras import losses as klosses

from RL.MEMORY import Memory
from RL.CONFIGS import DQN_CONFIG

class DQN():

    def __init__(self, memory, action_value: tf.keras.Model, metrics = [], config = DQN_CONFIG):
        self.config = config
        self.memory = memory
        self.step = 0
        self.last_action = None
        self.metrics = list(Metric(self) for Metric in metrics)

        self.action_value = action_value
        self.action_value_target = tf.keras.models.clone_model(action_value)
        self.opt = tf.keras.optimizers.Adam(1e-4)

        self.gamma = 0.99
        self.sample_size = 128
        self.frames_skipped = 1 
        self.history_lenght = 1 #To implement
        self.double_q_learning = False
        self.clipping = False #To implement
        self.reward_scaler = (0., 1.) #(mean, std), R <- (R-mean)/std
        self.update_method = "soft"
        
        self.train_freq = 1
        self.gradients_steps = 1
        self.target_update_interval = 1500
        self.tau = 0.99
        
        self.learning_starts = 2
        self.exploration_timesteps = 1000
        self.exploration_initial = 0.1
        self.exploration_final = 0.1
        self.f_eps = lambda s : max(s.exploration_final, s.exploration_initial + (s.exploration_final - s.exploration_initial) * (s.step / s.exploration_timesteps))

    def act(self, observation, greedy=False, mask = None):
        
        #Skip frames:
        if self.step % self.frames_skipped != 0:
            return self.last_action
        
        #Batching observation
        observations = tf.expand_dims(observation, axis = 0) # (1, observation_space)

        # Q(s)
        Q = self.action_value(observations) # (1, action_space)

        #Greedy policy
        epsilon = self.f_eps(self)
        if greedy or np.random.rand() > epsilon:
            if mask is not None:
                Q = Q - 10000.0 * tf.constant([mask], dtype = tf.float32)
            action = tf.argmax(Q, axis = -1, output_type = tf.int32).numpy()[0]

        #Exploration
        else :
            action = tf.random.uniform(shape = (1,), minval = 0, maxval = Q.shape[-1], dtype = tf.int32).numpy()[0]
            if mask is not None:
                authorized_actions = [i for i in range(len(mask)) if mask[i] == 0]
                action = random.choice(authorized_actions)
                
        # Action
        self.last_action = action
        return action


    def learn(self):
        metrics = list()
        
        #Skip frames:
        if self.step % self.frames_skipped != 0:
            return metrics

        #Learn only every train_freq steps
        self.step += 1
        if self.step % self.train_freq != 0:
            return metrics

        #Learn only after learning_starts steps 
        if self.step <= self.learning_starts:
            return metrics

        #Sample trajectories
        observations, actions, rewards, dones, next_observations = self.memory.sample(
            sample_size=self.sample_size,
            method = "random"
        )
        # print(observations, actions, rewards, dones)
        
        #Scaling the rewards
        if self.reward_scaler is not None:
            mean, std = self.reward_scaler
            rewards = (rewards - mean) / std
        
        # Estimated Q values
        if not self.double_q_learning:
            #Simple learning : Q(s,a) = R + max(Q_target(s_next))
            dones = tf.cast(dones, dtype = tf.float32)
            next_Q_values = tf.math.reduce_max(self.action_value_target(next_observations), axis = -1)
            expected_Q_values = rewards + self.gamma * next_Q_values * (1 - dones)
        else:
            #Double Q Learning : Q(s,a) = R + Q_target(s_next, argmax_a(Q(s_next, a)))
            dones = tf.cast(dones, dtype = tf.float32)
            actions_range = tf.range(len(actions))
            Q_values_s_next = self.action_value_target(next_observations)
            Q_values_s = self.action_value(next_observations)
            actions_indices = tf.argmax(Q_values_s, axis = -1)
            actions_indices = tf.cast(actions_indices, tf.int32)
            actions_indices = tf.stack((actions_range, actions_indices), axis = -1)
            next_Q_values = tf.gather_nd(Q_values_s_next, actions_indices)
            expected_Q_values = rewards + self.gamma * next_Q_values * (1 - dones)
        
        criterion = klosses.MeanSquaredError()
        for _ in range(self.gradients_steps):
            with tf.GradientTape() as tape:
                actions_range = tf.range(len(actions))
                actions_range = tf.reshape(actions_range, shape = (-1,1))
                actions_indices = tf.concat((actions_range, actions), axis = -1)
                Q_values_s = self.action_value(observations) 
                Q_values = tf.gather_nd(Q_values_s, actions_indices)
                loss = criterion(expected_Q_values, Q_values)

            
            #Batched Gradient descent
            gradients = tape.gradient(target = loss, sources = self.action_value.trainable_weights)
            self.opt.apply_gradients(zip(gradients, self.action_value.trainable_weights))

        #Update target every target_update_interval
        if self.update_method == "interval":
            if self.step % self.target_update_interval == 0:
                self.update_target_network()
        elif self.update_method == "soft":
            phi_target = np.array(self.action_value_target.get_weights(), dtype = object)
            phi = np.array(self.action_value.get_weights(), dtype= object)
            self.action_value_target.set_weights(self.tau * phi_target + (1-self.tau) * phi)    
        elif self.update_method == "periodic":
            if self.step % self.target_update_interval == 0:
                phi = np.array(self.action_value.get_weights(), dtype= object)
                self.action_value_target.set_weights(phi)
        
        else:
            print(f"Error : update_method {self.update_method} not implemented.")
            sys.exit()

        #Metrics
        return list(metric.on_learn(critic_loss = loss.numpy(), value = tf.reduce_mean(Q_values).numpy()) for metric in self.metrics)

    def remember(self, observation, action, reward, done, next_observation, info={}, **param):
        self.memory.remember((observation, action, reward, done, next_observation, info))
        return list(metric.on_remember(obs = observation, action = action, reward = reward, done = done, next_obs = next_observation) for metric in self.metrics)

    def update_target_network(self):
        self.action_value_target = tf.keras.models.clone_model(self.action_value)



if __name__ == "__main__":

    env = gym.make("CartPole-v0")

    action_value = tf.keras.models.Sequential([
        kl.Dense(16, activation='tanh'),
        kl.Dense(16, activation='tanh'),
        kl.Dense(env.action_space.n, activation='linear')
    ])

    MEMORY_KEYS = ['observation', 'action',
                       'reward', 'done', 'next_observation']
    memory = Memory(MEMORY_KEYS=MEMORY_KEYS, max_memory_len=40960)

    agent = DQN(memory=memory, action_value=action_value)  
    

    #sys.exit()



    episodes = 1000
    L_rewards_tot = list()
    L_loss = list()
    L_Q = list()
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

            if metrics is not None and metrics != list():
                print(metrics)
                raise
                L_loss.append(math.log(metrics["loss"]))
                L_Q.append(metrics["value"])
            reward_tot += reward

        L_rewards_tot.append(reward_tot)
        plt.clf()
        plt.plot(L_rewards_tot[-100:], label = "total reward")
        plt.plot(L_loss[-100:], label = "critic loss (log)")
        plt.plot(L_Q[-100:], label = "mean Q value")
        plt.legend()
        plt.show()
        plt.pause(1e-3)
        

    