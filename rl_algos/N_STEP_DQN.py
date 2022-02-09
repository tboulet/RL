import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import math
import gym
import sys
import matplotlib.pyplot as plt

kl = tf.keras.layers
ki = tf.keras.initializers

from MEMORY import Memory
from CONFIGS import DQN_CONFIG

class N_STEP_DQN():

    def __init__(self, memory, action_value: tf.keras.Model, n = 1, metrics = [], config = DQN_CONFIG):
        self.config = config
        self.memory = memory
        self.step = 1
        self.metrics = list(Metric(self) for Metric in metrics)

        self.action_value = action_value
        self.action_value_target = tf.keras.models.clone_model(action_value)
        self.opt = tf.keras.optimizers.Adam(1e-4)
        self.n = n

        self.gamma = 0.99
        self.sample_size = 4096

        self.train_freq = 10
        self.gradients_steps = 5
        self.target_update_interval = 1500
        self.tau = 0.99
        self.update_method = "soft"

        self.total_timesteps = 20000
        self.learning_starts = 0
        self.exploration_fraction = 0.3
        self.exploration_initial = 1
        self.exploration_final = 0.1
        self.f_eps = lambda s : max(s.exploration_final, s.exploration_initial + (s.exploration_final - s.exploration_initial) * (s.step / s.total_timesteps) / (s.exploration_fraction))

    def act(self, observation, greedy=False):

        observations = tf.expand_dims(observation, axis = 0) # (1, observation_space)

        # Q(s)
        Q = self.action_value(observations) # (1, action_space)

        #Greedy
        epsilon = self.f_eps(self)
        if greedy or np.random.rand() > epsilon:
            action = tf.argmax(Q, axis = -1, output_type = tf.int32)

        #Exploration
        else :
            action = tf.random.uniform(shape = (1,), minval = 0, maxval = Q.shape[-1], dtype = tf.int32)
        
        # Action
        return action.numpy()[0]


    def learn(self):
        metrics = list()

        #Learn only every train_freq steps
        self.step += 1
        if self.step % self.train_freq != 0:
            return metrics

        #Learn only after learning_starts steps 
        if self.step <= self.learning_starts:
            return metrics


        #Sampling trajectories
        observations, actions, rewards, dones, _ = self.memory.sample(  # (n, ?)
            sample_size = self.sample_size,
            method = 'last'
        )

        #Computing n step Q values estimate.
        L = rewards.shape[0]
        Q_values = tf.math.reduce_max(self.action_value_target(observations), axis = -1)
        expected_Q_values = list()

        for t in range(L):
            q_value = 0
            for t2 in range(t,L):     #Sum for n step.
                if dones[t2]:
                    q_value += rewards[t2] * self.gamma ** (t2-t)
                    break
                elif t2 - t == self.n:
                    q_value += Q_values[t2] * self.gamma ** self.n
                    break
                else:

                    q_value += rewards[t2] * self.gamma ** (t2-t)
            expected_Q_values.append(q_value)
            
        expected_Q_values = tf.convert_to_tensor(expected_Q_values)        


        #Gradient descent.
        for _ in range(self.gradients_steps):
            with tf.GradientTape() as tape:
                actions_range = tf.range(len(actions))
                actions_indices = tf.stack((actions_range, actions), axis = -1)
                Q_values_s = self.action_value(observations) 
                Q_values = tf.gather_nd(Q_values_s, actions_indices)
                loss = tf.keras.losses.mse(expected_Q_values, Q_values)
            
            #Batched Gradient descent
            gradients = tape.gradient(target = loss, sources = self.action_value.trainable_weights)
            self.opt.apply_gradients(zip(gradients, self.action_value.trainable_weights))

        #Update target every target_update_interval
        if self.update_method == "interval":
            if self.step % self.target_update_interval == 0:
                self.update_target_network()
                print("Target net updated.")
        elif self.update_method == "soft":
            phi_target = np.array(self.action_value_target.get_weights(), dtype = object)
            phi = np.array(self.action_value.get_weights(), dtype= object)
            self.action_value_target.set_weights(self.tau * phi_target + (1-self.tau) * phi)    
        else:
            print(f"Error : update_method {self.update_method} not implemented.")
            sys.exit()

        #Metrics
        return list(metric.on_learn(loss = loss.numpy(), value = tf.reduce_mean(Q_values).numpy()) for metric in self.metrics)

    def remember(self, observation, action, reward, done, next_observation, info={}, **param):
        self.memory.remember((observation, action, reward, done, next_observation, info))
        return list(metric.on_remember(obs = observation, action = action, reward = reward, done = done, next_obs = next_observation) for metric in self.metrics)

    def update_target_network(self):
        self.action_value_target = tf.keras.models.clone_model(self.action_value)


class N_STEP_DQN():
    
    def __init__(self, memory, action_value: tf.keras.Model, n = 1, metrics = [], config = DQN_CONFIG):
        self.config = config
        self.memory = memory
        self.step = 1
        self.metrics = list(Metric(self) for Metric in metrics)

        self.action_value = action_value
        self.action_value_target = tf.keras.models.clone_model(action_value)
        self.opt = tf.keras.optimizers.Adam(1e-4)
        self.n = n

        self.gamma = 0.99
        self.sample_size = 4096

        self.train_freq = 10
        self.gradients_steps = 5
        self.target_update_interval = 1500
        self.tau = 0.99
        self.update_method = "soft"

        self.total_timesteps = 20000
        self.learning_starts = 0
        self.exploration_fraction = 0.3
        self.exploration_initial = 1
        self.exploration_final = 0.1
        self.f_eps = lambda s : max(s.exploration_final, s.exploration_initial + (s.exploration_final - s.exploration_initial) * (s.step / s.total_timesteps) / (s.exploration_fraction))



    def act(self, observation, greedy=False):

        observations = tf.expand_dims(observation, axis = 0) # (1, observation_space)

        # Q(s)
        Q = self.action_value(observations) # (1, action_space)

        #Greedy
        epsilon = self.f_eps(self)
        if greedy or np.random.rand() > epsilon:
            action = tf.argmax(Q, axis = -1, output_type = tf.int32)

        #Exploration
        else :
            action = tf.random.uniform(shape = (1,), minval = 0, maxval = Q.shape[-1], dtype = tf.int32)
        
        # Action
        return action.numpy()[0]


    def learn(self):
        metrics = list()

        #Learn only every train_freq steps
        self.step += 1
        if self.step % self.train_freq != 0:
            return metrics

        #Learn only after learning_starts steps 
        if self.step <= self.learning_starts:
            return metrics


        #Sampling trajectories
        observations, actions, rewards, dones, _ = self.memory.sample(  # (n, ?)
            sample_size = self.sample_size,
            method = 'last'
        )

        #Computing n step Q values estimate.
        L = rewards.shape[0]
        Q_values = tf.math.reduce_max(self.action_value_target(observations), axis = -1)
        expected_Q_values = list()

        for t in range(L):
            q_value = 0
            for t2 in range(t,L):     #Sum for n step.
                if dones[t2]:
                    q_value += rewards[t2] * self.gamma ** (t2-t)
                    break
                elif t2 - t == self.n:
                    q_value += Q_values[t2] * self.gamma ** self.n
                    break
                else:

                    q_value += rewards[t2] * self.gamma ** (t2-t)
            expected_Q_values.append(q_value)
            
        expected_Q_values = tf.convert_to_tensor(expected_Q_values)        


        #Gradient descent.
        for _ in range(self.gradients_steps):
            with tf.GradientTape() as tape:
                actions_range = tf.range(len(actions))
                actions_indices = tf.stack((actions_range, actions), axis = -1)
                Q_values_s = self.action_value(observations) 
                Q_values = tf.gather_nd(Q_values_s, actions_indices)
                loss = tf.keras.losses.mse(expected_Q_values, Q_values)
            
            #Batched Gradient descent
            gradients = tape.gradient(target = loss, sources = self.action_value.trainable_weights)
            self.opt.apply_gradients(zip(gradients, self.action_value.trainable_weights))

        #Update target every target_update_interval
        if self.update_method == "interval":
            if self.step % self.target_update_interval == 0:
                self.update_target_network()
                print("Target net updated.")
        elif self.update_method == "soft":
            phi_target = np.array(self.action_value_target.get_weights(), dtype = object)
            phi = np.array(self.action_value.get_weights(), dtype= object)
            self.action_value_target.set_weights(self.tau * phi_target + (1-self.tau) * phi)    
        else:
            print(f"Error : update_method {self.update_method} not implemented.")
            sys.exit()

        #Metrics
        return list(metric.on_learn(loss = loss.numpy(), value = tf.reduce_mean(Q_values).numpy()) for metric in self.metrics)

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

    agent = N_STEP_DQN(memory=memory, action_value=action_value)  
    

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

            if metrics is not None:
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
        

    