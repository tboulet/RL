from copy import copy, deepcopy
from operator import index
import numpy as np
import math
import gym
import sys
import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from MEMORY import Memory
from CONFIGS import DQN_CONFIG

class DQN():

    def __init__(self, memory, action_value : nn.Module, metrics = [], config = DQN_CONFIG):
        self.config = config
        self.memory = memory
        self.step = 0
        self.last_action = None
        
        self.action_value = action_value
        self.action_value_target = deepcopy(action_value)
        self.opt = optim.Adam(lr = 1e-4, params=action_value.parameters())
        self.opt = optim.Adam(params = action_value.parameters(), lr=1e-4)

        self.gamma = 0.99
        self.sample_size = 512
        self.frames_skipped = 1 
        self.history_lenght = 1 #To implement
        self.double_q_learning = False
        self.clipping = 10
        self.reward_scaler = (0, 500) #(mean, std), R <- (R-mean)/std
        self.update_method = "soft"
        
        self.train_freq = 4
        self.gradients_steps = 4
        self.target_update_interval = 5000
        self.tau = 0.99
        
        self.learning_starts = 2048
        self.exploration_timesteps = 10000
        self.exploration_initial = 1
        self.exploration_final = 0.05
        self.f_eps = lambda s : max(s.exploration_final, s.exploration_initial + (s.exploration_final - s.exploration_initial) * (s.step / s.exploration_timesteps))
        self.metrics = list(Metric(self) for Metric in metrics)
        
        
    def act(self, observation, greedy=False, mask = None):
        '''Ask the agent to take a decision given an observation.
        observation : an (n_obs,) shaped observation.
        greedy : whether the agent always choose the best Q values according to himself.
        mask : a binary list containing 1 where corresponding actions are forbidden.
        return : an int corresponding to an action
        '''

        #Skip frames:
        if self.step % self.frames_skipped != 0:
            return self.last_action
        
        #Batching observation
        observations = torch.Tensor(observation)
        observations = observations.unsqueeze(0) # (1, observation_space)
    
        # Q(s)
        Q = self.action_value(observations) # (1, action_space)

        #Greedy policy
        epsilon = self.f_eps(self)
        if greedy or np.random.rand() > epsilon:
            with torch.no_grad():
                if mask is not None:
                    Q = Q - 10000.0 * torch.Tensor([mask])      # So that forbidden action won't ever be selected by the argmax.
                action = torch.argmax(Q, axis = -1).numpy()[0]  
    
        #Exploration
        else :
            if mask is None:
                action = torch.randint(size = (1,), low = 0, high = Q.shape[-1]).numpy()[0]     #Choose random action
            else:
                authorized_actions = [i for i in range(len(mask)) if mask[i] == 0]              #Choose random action among authorized ones
                action = random.choice(authorized_actions)
    
        # Action
        self.last_action = action
        return action


    def learn(self):
        '''Do one step of learning.
        return : metrics, a list of metrics computed during this learning step.
        '''
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
            method = "random",
            func = lambda arr : torch.Tensor(arr),
        )
        actions = actions.to(dtype = torch.int64)
        #print(observations, actions, rewards, dones, sep = '\n\n')
    

        #Scaling the rewards
        if self.reward_scaler is not None:
            mean, std = self.reward_scaler
            rewards = (rewards - mean) / std
        
        # Estimated Q values
        if not self.double_q_learning:
            #Simple learning : Q(s,a) = r + gamma * max_a'(Q_target(s_next, a')) * (1-d)  | s_next and r being the result of action a taken in observation s
            future_Q_s_a = self.action_value_target(next_observations)
            future_Q_s, bests_a = torch.max(future_Q_s_a, dim = 1, keepdim=True)
            Q_s_predicted = rewards + self.gamma * future_Q_s * (1 - dones)  #(n_sampled,)
        else:
            #Double Q Learning : Q(s,a) = r + gamma * Q_target(s_next, argmax_a'(Q(s_next, a')))
            future_Q_s_a = self.action_value(next_observations)
            future_Q_s, bests_a = torch.max(future_Q_s_a, dim = 1, keepdim=True)
            future_Q_s_a_target = self.action_value_target(next_observations)
            future_Q_s_target = torch.gather(future_Q_s_a_target, dim = 1, index= bests_a)
            Q_s_predicted = rewards + self.gamma * future_Q_s_target * (1 - dones)
            # print(future_Q_s_a.shape, bests_a.shape, future_Q_s_a_target.shape, future_Q_s_target.shape, Q_s_predicted.shape, 
            #       )
            # raise        
        
        #Gradient descent on Q network
        criterion = nn.SmoothL1Loss()
        for _ in range(self.gradients_steps):
            self.opt.zero_grad()
            Q_s_a = self.action_value(observations)
            Q_s = Q_s_a.gather(dim = 1, index = actions)
            loss = criterion(Q_s_predicted, Q_s)
            loss.backward(retain_graph = True)
            if self.clipping is not None:
                for param in self.action_value.parameters():
                    param.grad.data.clamp_(-self.clipping, self.clipping)
            self.opt.step() 
        
        #Update target network
        if self.update_method == "periodic":
            if self.step % self.target_update_interval == 0:
                self.update_target_network()
        elif self.update_method == "soft":
            for phi, phi_target in zip(self.action_value.parameters(), self.action_value_target.parameters()):
                phi.data = self.tau * phi.data + (1-self.tau) * phi_target.data
            
        else:
            print(f"Error : update_method {self.update_method} not implemented.")
            sys.exit()

        #Metrics
        return list(metric.on_learn(critic_loss = loss.detach().numpy(), value = Q_s.mean().detach().numpy()) for metric in self.metrics)

    def remember(self, observation, action, reward, done, next_observation, info={}, **param):
        self.memory.remember((observation, action, reward, done, next_observation, info))
        return list(metric.on_remember(obs = observation, action = action, reward = reward, done = done, next_obs = next_observation) for metric in self.metrics)

    def update_target_network(self):
        self.action_value_target = deepcopy(self.action_value)



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
        

    