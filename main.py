#Gym for environments, WandB for feedback
import wandb


def run(agent, 
        env, 
        steps = float('inf'), 
        episodes = float('inf'), 
        wandb_cb = False, 
        n_render = 20
        ):
    '''Train an agent on an env.
    agent : an AGENT instance (with methods act, learn and remember implemented)
    env : a gym env (with methods reset, step, render)
    steps : int, number of steps of training
    episodes : int, number of maximal episodes maximal for training
    wandb_cb : bool, whether metrics are logged in WandB
    n_render : int, one episode on n_render is rendered
    '''
    
    print("Run starts.")
################### FEEDBACK #####################
    if n_render == None: n_render = float('inf')
    if wandb_cb: 
        try:
            from config import project, entity
        except ImportError:
            raise Exception("You need to specify your WandB ids in config.py\nConfig template is available at div/config_template.py")
        run = wandb.init(project=project, 
                        entity=entity,
                        config=agent.config,
        )
##################### END FEEDBACK ###################
    episode = 1
    step = 0
    while step < steps:
        done = False
        obs = env.reset()
        
        
        while not done and step < steps and episode < episodes:
            action = agent.act(obs)                                                 #Agent acts
            next_obs, reward, done, info = env.step(action)                         #Env reacts            
            agent.remember(obs, action, reward, done, next_obs, info)    #Agent saves previous transition in its memory
            agent.learn()                                                #Agent learn (eventually)
            
            ###### Feedback ######
            print(f"Episode n°{episode} - Total step n°{step} ...", end = '\r')
            if episode % n_render == 0:
                env.render()
            if wandb_cb:
                agent.log_metrics()
            ######  End Feedback ######  

            #If episode ended, reset env, else change state
            if done:
                step += 1
                episode += 1
                break
            else:
                step += 1
                obs = next_obs
    
    if wandb_cb: run.finish()   #End wandb run.
    print("End of run.")
    
    
    

if __name__ == "__main__":
    #Import config
    try:
        from config import episodes, steps, wandb_cb, n_render
    except ImportError:
        raise Exception("You need to specify your config in config.py\nConfig template is available at div/config_template.py")
    
    #ENVIRONMENT
    from RL.ENV import create_env
    env = create_env()
    
    #FUNCTION APPROXIMATION
    from RL.FA import create_functions
    functions = create_functions(env)
    
    #AGENT
    from RL.RL_AGENTS import create_agent
    agent = create_agent(functions)
    
    
    #RUN
    run(agent, 
        env = env, 
        steps = steps, 
        episodes = episodes,
        wandb_cb = wandb_cb,
        n_render = n_render,
        )
    