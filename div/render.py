import gym

def render_agent(agent, env = gym.make("CartPole-v0"), episodes = 10, show_metrics = False):
    
    for episode in range(episodes):
        done = False
        obs = env.reset()

        while not done:
            action = agent.act(obs)
            next_obs, reward, done, info = env.step(action)
            env.render()
            metrics1 = agent.remember(obs, action, reward, done, next_obs, info)
            metrics2 = agent.learn()

            if show_metrics:
                print("\n\tMETRICS : ")
                for metrics in metrics1 + metrics2:
                    for key, value in metrics.items():
                        print(f"{key}: {value}")

            #If episode ended.
            if done:
                pass
            else:
                obs = next_obs