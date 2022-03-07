from abc import ABC, abstractmethod
import torch
import wandb
from random import randint
from RL.METRICS import *
from div.utils import pr_and_raise, pr_shape

class AGENT(ABC):
    
    def __init__(self, config = dict(), metrics = list()):
        self.step = 0
        self.episode = 0
        self.metrics = [Metric(self) for Metric in metrics]
        self.config = config
        for name, value in config.items():
            setattr(self, name, value)
        self.metrics_saved = list()
        
    @abstractmethod
    def act(self, obs):
        pass
    
    @abstractmethod
    def learn(self):
        pass

    @abstractmethod
    def remember(self, **kwargs):
        pass
    
    def add_metric(self, mode, **values):
        if mode == 'act':
            for metric in self.metrics:
                self.metrics_saved.append(metric.on_act(**values))
        if mode == 'remember':
            for metric in self.metrics:
                self.metrics_saved.append(metric.on_remember(**values))
        if mode == 'learn':
            for metric in self.metrics:
                self.metrics_saved.append(metric.on_learn(**values))
    
    def log_metrics(self):
        for metric in self.metrics_saved:
            wandb.log(metric, step = self.step)
        self.metrics_saved = list()
    
    
    def compute_SARSA(self, rewards, next_observations, next_actions, dones, Q_scalar = True, model = 'action_value'):
        '''Compute the 1 step TD estimates V(s) of state values.
        observations, actions, rewards, next_observations, next_actions : (T, *dims) shaped torch tensors
        model : the name of the attribute of agent used for computing state values, in ('action_value', 'action_value_target')
        return : a (T, 1) shaped torch tensor representing state values
        '''
        model = getattr(self, model)
        
        if Q_scalar :
            Q_s_future = model(next_observations, next_actions)
        else:
            raise
        
        return rewards + (1 - dones) * self.gamma * Q_s_future
    
    
    def compute_SARSA_n_step(self, rewards, next_observations, next_actions, dones):
        '''Compute the 1 step TD estimates V(s) of state values.
        observations, actions, rewards, next_observations, next_actions : (T, *dims) shaped torch tensors
        model : the name of the attribute of agent used for computing state values, in ('action_value', 'action_value_target')
        return : a (T, 1) shaped torch tensor representing state values
        '''
        raise #To implement
    
    
    def compute_TD(self, observations, actions, rewards, next_observations, dones, model = 'state_value'):
        '''Compute the 1 step TD estimates V(s) of state values.
        rewards : a (T, 1) shaped torch tensor representing rewards
        observations : a (T, *dims) shaped torch tensor representing observations
        model : the name of the attribute of agent used for computing state values, in ('state_value', 'state_value_target')
        return : a (T, 1) shaped torch tensor representing state values
        '''
        model = getattr(self, model)
        raise   #test phase
        return rewards + (1 - dones) * self.gamma * model(next_observations)
    
    
    def compute_TD_n_step(self, rewards, observations, model = "state_value"):
        '''Compute the n_step TD estimates V(s) of state values over one episode, where n_step is an int attribute of agent.
        It follows the Temporal Difference relation V(St) = Rt + g*Rt+1 + ... + g^n-1 * Rt+n-1 + g^n * V(St+n)
        rewards : a (T, 1) shaped torch tensor representing rewards
        observations : a (T, *dims) shaped torch tensor representing observations
        model : the name of the attribute of agent used for computing state values, in ('state_value', 'state_value_target')
        return : a (T, 1) shaped torch tensor representing state values
        '''
        n = self.n_step
        
        #We compute the discounted sum of the n next rewards dynamically.
        T = len(rewards)
        rewards = rewards[:, 0]
        n_next_rewards =  [0 for _ in range(T)] + [0]
        t = T - 1
        while t >= 0:   
            if t >= T - n:
                n_next_rewards[t] = rewards[t] + self.gamma * n_next_rewards[t+1]
            else:
                n_next_rewards[t] = rewards[t] + self.gamma * n_next_rewards[t+1] - (self.gamma ** n) * rewards[t+n]
            t -= 1
        n_next_rewards.pop(-1)
        n_next_rewards = torch.Tensor(n_next_rewards).unsqueeze(-1)

        #We compute the state value, and shift them forward in order to add them or not to the estimate.
        model = getattr(self, model)
        state_values = model(observations)
        state_values_to_add = torch.concat((state_values, torch.zeros(n, 1)), axis = 0)[n:]
        
        V_targets = n_next_rewards + state_values_to_add
        return V_targets    
        
        
    def compute_MC(self, rewards):
        '''Compute the sums of future rewards (discounted) over one episode.
        It is the Monte Carlo estimation of a state value : Rt + g * Rt+1 + ... + g^T-t * RT
        rewards : a (T, 1) shaped torch tensor representing rewards
        return : a (T, 1) shaped torch tensor representing discounted sum of future rewards
        '''
        #We compute the discounted sum of the next rewards dynamically.
        T = len(rewards)
        rewards = rewards[:, 0]
        future_rewards =  [None for _ in range(T)] + [0]
        t = T - 1
        while t >= 0:   
            future_rewards[t] = rewards[t] + self.gamma * future_rewards[t+1]
            t -= 1
        future_rewards.pop(-1)
        future_rewards = torch.Tensor(future_rewards).unsqueeze(-1)          
        return future_rewards
    
    
    def compute_GAE(self, rewards, observations):
        '''Compute the Generalized Advantage Estimator (GAE) of advantage function over one episode.
        rewards : a (T, 1) shaped torch tensor representing rewards
        observations : a (T, *dims) shaped torch tensor representing observations
        return : a (T, 1) shaped torch tensor representing advantages functions
        '''
        T = len(rewards)
        rewards = rewards[:, 0].numpy()
        values = self.state_value(observations)[:, 0].detach().numpy()
        #We compute the TD residuals delta_t = Rt + (1-Dt) * g * V(St+1) - V(St)
        deltas = list()
        for t in range(T-1):
            deltas.append(rewards[t] + self.gamma * values[t+1] - values[t])
        deltas.append(rewards[T-1] - values[T-1])   #Last residual is just RT - V(ST) since this is the end of episode.
        #We compute dynamically the GAE
        A_GAE = [None for _ in range(T)] + [0]
        t = T - 1
        while t >= 0:
            A_GAE[t] = deltas[t] + self.gamma * self.gae_lambda * A_GAE[t + 1]
            t -= 1
        A_GAE.pop(-1)
        A_GAE = torch.Tensor(A_GAE).unsqueeze(-1)                
        return A_GAE
       

#Use the following agent as a model for minimum restrictions on AGENT subclasses :
class RANDOM_AGENT(AGENT):
    '''A random agent evolving in a discrete environment.
    n_actions : int, n of action space
    '''
    def __init__(self, n_actions):
        super().__init__(metrics=[MetricS_On_Learn_Numerical, Metric_Performances]) #Choose metrics here
        self.n_actions = n_actions  #For RandomAgent only
    
    def act(self, obs):
        #Choose action here
        ...
        action = randint(0, self.n_actions - 1)
        #Save metrics
        values = {"my_metric_name1" : 22, "my_metric_name2" : 42}
        self.add_metric(mode = 'act', **values)
        
        return action
    
    def learn(self):
        #Learn here
        ...
        #Save metrics
        self.step += 1
        values = {"my_metric_name1" : 22, "my_metric_name2" : 42}
        self.add_metric(mode = 'learn', **values)
    
    def remember(self, *args):
        #Save kwargs in memory here
        ... 
        #Save metrics
        values = {"my_metric_name1" : 22, "my_metric_name2" : 42}
        self.add_metric(mode = 'remember', **values)