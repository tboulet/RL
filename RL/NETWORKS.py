# Generate networks (or other objects) adapted to the environment and to the function of the network.
import torch.nn as nn
import gym.spaces as spaces

def create_networks(env):
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
        
    else:
        raise Exception("Unknow type of gym env.")
        
    return {"actor" : actor,
            "state_value" : state_value,
            "action_value" : action_value,
            }