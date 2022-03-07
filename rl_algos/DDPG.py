from copy import copy, deepcopy
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from div.utils import *
from RL.MEMORY import Memory
from RL.CONFIGS import DDPG_CONFIG
from RL.METRICS import *
from rl_algos.AGENT import AGENT

class DDPG(AGENT):
    '''DDPG
    '''

    def __init__(self, actor : nn.Module, action_value : nn.Module):
        metrics = [Metric_Total_Reward, MetricS_On_Learn]
        super().__init__(config = DDPG_CONFIG, metrics = metrics)
        self.memory = Memory(MEMORY_KEYS = ['observation', 'action','reward', 'done', 'next_observations'])
        
        self.action_value = action_value
        self.action_value_target = deepcopy(action_value)
        self.opt_critic = optim.Adam(lr = self.learning_rate_critic, params=self.action_value.parameters())
        
        self.policy = actor
        self.policy_target = deepcopy(actor)
        self.opt_policy = optim.Adam(lr = self.learning_rate_actor, params=self.policy.parameters())
                        
        
    def act(self, observation, mask = None):
        '''Ask the agent to take a decision given an observation.
        observation : an (n_obs,) shaped numpy observation.
        mask : a binary list containing 1 where corresponding actions are forbidden.
        return : an (dim_action,) shaped action
        '''
        
        #Batching observation
        observation = torch.Tensor(observation)
        observations = observation.unsqueeze(0) # (1, observation_space)
        #Choose action
        action = self.policy(observations).detach()[0]
        n_actions = action.shape[0]
        noise = torch.normal(mean = 0, std=torch.ones((n_actions,)) * self.sigma)
        action = (action + noise).numpy()
        
        #Save metrics
        self.add_metric(mode = 'act')

        # Action
        return action


    def learn(self):
        '''Do one step of learning.
        '''
        values = dict()
        self.step += 1
        
        #Learn every train_freq episode
        if self.step % self.train_freq != 0:
            return
        
        #Sample trajectories
        observations, actions, rewards, dones, next_observations = self.memory.sample(
            sample_size=self.sample_size,
            method = "random",
            )
        
        #Compute targets of Q using SARSA
        next_actions = self.policy_target(next_observations)
        Q_s_targets = self.compute_SARSA(rewards, next_observations, next_actions, dones, 
                                         Q_scalar=True, 
                                         model="action_value_target",
                                            ).detach()
        
        #Update critic by minimizing the loss
        Q_s = self.action_value(observations, actions)
        criterion = nn.MSELoss()
        critic_loss = criterion(Q_s, Q_s_targets).mean()
        
        self.opt_critic.zero_grad()
        critic_loss.backward()
        self.opt_critic.step()
        
        #Update actor by maximizing Q values
        actions = self.policy(observations)
        Q_s_to_maximize = self.action_value(observations, actions)
        actor_loss = - Q_s_to_maximize.mean()  
        
        self.opt_policy.zero_grad()
        actor_loss.backward()
        self.opt_policy.step()
        
        #Update target networks
        for phi, phi_target in zip(self.action_value.parameters(), self.action_value_target.parameters()):
            phi_target.data = self.tau * phi_target.data + (1-self.tau) * phi.data 
        for theta, theta_target in zip(self.policy.parameters(), self.policy_target.parameters()):
            theta_target.data = self.tau * theta_target.data + (1-self.tau) * theta.data 
        
        #Save metrics
        values["critic_loss"] = critic_loss.detach().numpy()
        values["actor_loss"] = actor_loss.detach().numpy()
        values["value"] = Q_s_to_maximize.detach().mean().numpy()
        self.add_metric(mode = 'learn', **values)
        
        
    def remember(self, observation, action, reward, done, next_observation, info={}, **param):
        '''Save elements inside memory.
        *arguments : elements to remember, as numerous and in the same order as in self.memory.MEMORY_KEYS
        '''
        self.memory.remember((observation, action, reward, done, next_observation, info))
            
        #Save metrics
        values = {"obs" : observation, "action" : action, "reward" : reward, "done" : done}
        self.add_metric(mode = 'remember', **values)