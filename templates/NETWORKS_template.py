# Generate networks (or other objects) adapted to the environment and to the function of the network.
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym.spaces as spaces

def create_networks(env):
    # State space continuous but action are discrete
    if isinstance(env.action_space, spaces.Discrete) and isinstance(env.observation_space, spaces.Box):
        print(f"\nCreation of networks for gym environment {env}. Type of spaces :\nS = CONTINUOUS\nA = DISCRETE\n")
    
        n_obs, *args = env.observation_space.shape
        n_actions = env.action_space.n

        #ACTOR PI
        actor = nn.Sequential(
                nn.Linear(n_obs, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, n_actions),
                nn.Softmax(dim=-1),
            )

        #CRITIC Q
        action_value = nn.Sequential(
                nn.Linear(n_obs, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, n_actions),
            )

        #STATE VALUE V
        state_value = nn.Sequential(
                nn.Linear(n_obs, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
        
    # State space is continuous but action are also continuous here
    elif isinstance(env.action_space, spaces.Box) and isinstance(env.observation_space, spaces.Box):
        print(f"\nCreation of networks for gym environment {env}. Type of spaces :\nS = CONTINUOUS\nA = CONTINUOUS\n")
    
        n_obs, *args = env.observation_space.shape
        dim_actions = len(env.action_space.shape)
        
        #ACTOR PI
        class Actor_continuous(nn.Module):
            def __init__(self):
                super(Actor_continuous, self).__init__()
                a_high = env.action_space.high
                a_low = env.action_space.low
                self.range_action = torch.Tensor(a_high - a_low)
                self.mean_action = torch.Tensor((a_high + a_low)/2)
                self.fc1 = nn.Linear(n_obs, 64)
                self.fc2 = nn.Linear(64, 64)
                self.fc3 = nn.Linear(64, dim_actions)
            def forward(self, x):
                x = F.relu(self.fc1(x))
                # x = F.relu(self.fc2(x))
                x = torch.tanh(self.fc3(x))                
                action = x * self.range_action / 2 + self.mean_action
                return action
        actor = Actor_continuous()

        #CRITIC Q
        class Action_value_continuous(nn.Module):
            def __init__(self):
                super(Action_value_continuous, self).__init__()
                self.fc_obs1 = nn.Linear(n_obs, 32)
                self.fc_obs2 = nn.Linear(32, 32)
                
                self.fc_action1 = nn.Linear(dim_actions, 32)
                self.fc_action2 = nn.Linear(32, 32)
                
                self.fc_global1 = nn.Linear(64, 32)
                self.fc_global2 = nn.Linear(32, 1)
            def forward(self, s, a):
                s = F.relu(self.fc_obs1(s))
                # s = F.relu(self.fc_obs2(s))
                a = F.relu(self.fc_action1(a))
                # a = F.relu(self.fc_action2(a))
                sa = torch.concat([s,a], dim = -1)
                sa = F.relu(self.fc_global1(sa))
                sa = self.fc_global2(sa)
                return sa
        action_value = Action_value_continuous()

        #STATE VALUE V
        state_value = None
        
    else:
        raise Exception("Unknow type of gym env.")
        
    return {"actor" : actor,
            "state_value" : state_value,
            "action_value" : action_value,
            "Q_table" : None,
            }