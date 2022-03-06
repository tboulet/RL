from abc import ABC, abstractmethod
import torch
import wandb
from random import randint
from METRICS import *
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
        
    def compute_TD(self, rewards, observations, model = "state_value"):
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
        future_rewards =  [0 for _ in range(T)] + [0]
        t = T - 1
        while t >= 0:   
            future_rewards[t] = rewards[t] + self.gamma * future_rewards[t+1]
            t -= 1
        future_rewards.pop(-1)
        future_rewards = torch.Tensor(future_rewards).unsqueeze(-1)          
        return future_rewards
    
       

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