from copy import copy, deepcopy
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

from RL.MEMORY import Memory, Memory_episodic
from RL.CONFIGS import DQN_CONFIG
from RL.METRICS import *
from div.utils import pr_shape
from rl_algos.AGENT import AGENT

class N_DQN(AGENT):

    def __init__(self, action_value : nn.Module):
        metrics = [MetricS_On_Learn, Metric_Total_Reward, Metric_Epsilon]
        super().__init__(config = DQN_CONFIG, metrics = metrics)
        self.memory = Memory_episodic(MEMORY_KEYS = ['observation', 'action','reward', 'done', 'prob'])
        
        self.action_value = action_value
        self.action_value_target = deepcopy(action_value)
        self.opt = optim.Adam(lr = self.learning_rate, params=action_value.parameters())
        self.f_eps = lambda s : max(s.exploration_final, s.exploration_initial + (s.exploration_final - s.exploration_initial) * (s.step / s.exploration_timesteps))
        
    def act(self, observation, mask = None, training = True):
        '''Ask the agent to take a decision given an observation.
        observation : an (n_obs,) shaped observation.
        mask : a binary list containing 1 where corresponding actions are forbidden.
        return : an int corresponding to an action
        '''

        #Batching observation
        observations = torch.Tensor(observation)
        observations = observations.unsqueeze(0) # (1, observation_space)
    
        # Q(s)
        Q = self.action_value(observations) # (1, action_space)
        action_greedy = torch.argmax(Q, axis = -1).detach().numpy()[0] 
        _, n_actions = Q.shape
        
        #Epsilon-greedy policy
        epsilon = self.f_eps(self)
        if np.random.rand() > epsilon:
            prob = 1-epsilon + epsilon/n_actions
            action = action_greedy
    
        else :
            action = torch.randint(size = (1,), low = 0, high = Q.shape[-1]).numpy()[0]     #Choose random action
            if action == action_greedy:
                prob = 1-epsilon + epsilon/n_actions
            else:
                prob = epsilon/n_actions
        
        #Save metrics
        if training: 
            self.last_prob = prob
            self.add_metric(mode = 'act')
    
        # Action
        return action


    def learn(self):
        '''Do one step of learning.
        '''
        
        values = dict()
        self.step += 1

        #Learn only at the end of episodes, and only every train_freq_episodes episodes.
        if not self.memory.done:
            return
        self.episode += 1
        if self.episode % 1 != 0:
            return

        #Sample trajectories
        episodes = self.memory.sample(
            sample_size=self.sample_size,
            method = "random",
            )

        Q_targets = list()
        for observations, actions, rewards, dones, probs in episodes:
            #Scaling the rewards
            if self.reward_scaler is not None:
                rewards = rewards / self.reward_scaler
            #Type errors
            actions = actions.to(dtype = torch.int64)
            #Compute Q targets
            next_observations = torch.roll(observations, shifts = -1, dims = 0)
            
            # #SARSA : E_mu[Rt + g * (1-Dt) * Q(St+1, At+1)]
            # next_actions = torch.roll(actions, shifts = -1, dims = 0)
            # print(next_actions.shape)
            # q_targets = self.compute_SARSA(rewards, next_observations, next_actions, dones, 
            #                                model = 'action_value',
            #                                importance_weights = None,
            #                                )
            
            # #SARSA unbiased : E_mu[Rt + g * (1-Dt) * r_t+1 * Q(St+1, At+1)]
            # next_actions = torch.roll(actions, shifts = -1, dims = 0)
            # best_actions_for_Q = torch.argmax(self.action_value(observations), dim = 1, keepdim=True)
            # greedyQ_probs = (actions == best_actions_for_Q).float()
            # importance_weights = greedyQ_probs / probs
            # q_targets = self.compute_SARSA(rewards, next_observations, next_actions, dones, 
            #                                model = 'action_value',
            #                                importance_weights = importance_weights,
            #                                )
                        
            # #SARSA Expected : E_mu[Rt + g * (1-Dt) * Q(St+1, pi(St+1))]
            # q_targets = self.compute_SARSA(rewards, next_observations, dones, 
            #                                model = 'action_value',
            #                                importance_weights = None,
            #                                )
            
            #SARSAN : E_mu[Rt + g * Rt+1 + g² * Rt+2 + g^3 * Q(St+3, At+3)]
            q_targets = self.compute_SARSA_n_step(rewards, observations, actions, model = 'action_value', importance_weights=None)
                                           
            
            #SARSAN unbiased : E_mu[Rt + g * Rt+1 + g² * Rt+2 + g^3 * Q(St+3, At+3)]
            
            
            #SARSAN unbiased 
            
            #SARSAN
            
            Q_targets.append(q_targets)
            
            
        observations, actions, rewards, dones, probs = [torch.concat([episode[elem] for episode in episodes], axis = 0) for elem in range(len(episodes[0]))]
        Q_targets = torch.concat(Q_targets, axis = 0).detach()
        
        #Type errors...
        actions = actions.to(dtype = torch.int64)
        
        #Scaling the rewards
        if self.reward_scaler is not None:
            rewards = rewards / self.reward_scaler
            
        #Gradient descent on Q network
        criterion = torch.nn.MSELoss()
        for _ in range(self.gradients_steps):
            self.opt.zero_grad()
            Q_s = self.QSA(self.action_value, observations, actions)
            loss = criterion(Q_s, Q_targets)
            loss.backward(retain_graph = True)
            self.opt.step()
        
        #Update target network
        for phi, phi_target in zip(self.action_value.parameters(), self.action_value_target.parameters()):
            phi_target.data = self.tau * phi_target.data + (1-self.tau) * phi.data    
       
        #Save metrics*
        values["critic_loss"] = loss.detach().numpy()
        values["value"] = Q_s.mean().detach().numpy()
        self.add_metric(mode = 'learn', **values)
        
        
    def remember(self, observation, action, reward, done, next_observation, info={}, **param):
        '''Save elements inside memory.
        *arguments : elements to remember, as numerous and in the same order as in self.memory.MEMORY_KEYS
        '''
        prob = self.last_prob
        self.memory.remember((observation, action, reward, done, prob))        
        
        #Save metrics
        values = {"obs" : observation, "action" : action, "reward" : reward, "done" : done}
        self.add_metric(mode = 'remember', **values)
    
