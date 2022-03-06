import gym

def create_env():
    try:
        from config import env_name
    except ImportError:
        raise Exception("You need to specify gym environment name in config.py, example 'CartPole-v0'")
    env = gym.make(env_name)
    return env