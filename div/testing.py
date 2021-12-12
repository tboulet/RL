import sys
import gym

def test_agent(agent, env = gym.make("CartPole-v0")):

    print("Test 1 : act")
    obs = env.reset()
    try:
        for _ in range(5):
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
        print("Test 1 passé.")
    except Exception as E:
        print(E)
        print("Test 1 échoué.")
        sys.exit()


    print("Test 2 : remember")
    try:
        obs = env.reset()
        for _ in range(5):
            action = agent.act(obs)
            next_obs, reward, done, _ = env.step(action)
            agent.remember(obs, action, reward, done, next_obs)
            traj = agent.memory.sample(2, 1)
            obs = next_obs
    except Exception as E:
        print(E)
        print("Test 3 échoué.")
        sys.exit()
    print("Test 2 passé.")


    print("Test 3 : learn")
    try:
        agent.learn()
        print("Test 3 passé.")
    except Exception as E:
        print(E)
        print("Test 3 échoué.")
        sys.exit()







