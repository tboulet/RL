from rl_algos.AGENT import RANDOM_AGENT
from rl_algos.REINFORCE import REINFORCE, REINFORCE_OFFPOLICY
from rl_algos.DQN import DQN
from rl_algos.N_DQN import N_DQN
from rl_algos.ACTOR_CRITIC import ACTOR_CRITIC
from rl_algos.PPO import PPO
from rl_algos.DDPG import DDPG


def create_agent(networks):
    try:
        from config import agent_name
    
        if agent_name == 'dqn':
            action_value = networks['action_value']
            agent = DQN(action_value)
            
        elif agent_name == 'n_dqn':
            action_value = networks['action_value']
            agent = N_DQN(action_value)
        
        elif agent_name == 'reinforce':
            actor = networks['actor']
            agent = REINFORCE(actor)
        
        elif agent_name == 'reinforce_offpolicy':
            actor = networks['actor']
            agent = REINFORCE_OFFPOLICY(actor)
            
        elif agent_name == 'ac':
            actor = networks['actor']
            state_value = networks['state_value']
            action_value = networks['action_value']
            agent = ACTOR_CRITIC(actor, action_value, state_value)
            
        elif agent_name == 'ppo':
            actor = networks['actor']
            state_value = networks['state_value']
            agent = PPO(actor, state_value)
            
        elif agent_name == 'ddpg':
            actor = networks['actor']
            action_value = networks['action_value']
            agent = DDPG(actor, action_value)
    
        return agent
    
    except ImportError:
        raise Exception("You need to specify your agent name in config.py\nConfig template is available at div/config_template.py")
    
    