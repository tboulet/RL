class Metric():
    def __init__(self):
        pass

    def on_learn(self, **kwargs):
        return dict()
    
    def on_remember(self, **kwargs):
        return dict()


class Metric_Total_Reward(Metric):
    
    def __init__(self, agent):
        super().__init__()
        self.agent = agent    
        self.total_reward = 0
        self.new_episode = False

    def on_remember(self, **kwargs):
        try:
            if self.new_episode: 
                self.total_reward = 0
                self.new_episode = False
            self.total_reward += kwargs["reward"]

            if kwargs["done"]:
                self.new_episode = True
                return {"total_reward" : self.total_reward}
            else:
                return dict()
        except KeyError:
            return dict()
    

class Metric_Epsilon(Metric):

    def __init__(self, agent):
        super().__init__()
        self.agent = agent

    def on_learn(self, **kwargs):
        try:
            return {"epsilon" : self.agent.f_eps(self.agent)}
        except:
            return dict()



class Metric_Actor_Loss(Metric):
    def __init__(self, agent):
        super().__init__()
        self.agent = agent
    
    def on_learn(self, **kwargs):
        try:
            return {"actor_loss" : kwargs["actor_loss"]}
        except KeyError:
            return dict()


class Metric_Critic_Loss(Metric):
    def __init__(self, agent):
        super().__init__()
        self.agent = agent
        
    def on_learn(self, **kwargs):
        try:
            return {"critic_loss" : kwargs["critic_loss"]}
        except KeyError:
            return dict()


class Metric_Critic_Value(Metric):
    def __init__(self, agent):
        super().__init__()
        self.agent = agent
        
    def on_learn(self, **kwargs):
        try:
            return {"value" : kwargs["value"]}
        except KeyError:
            return dict()