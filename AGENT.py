

class AGENT:
    
    def __init__(self, config):
        for name, value in config.items():
            setattr(self, name, value)
            
    def act(self):
        pass
    
    def learn(self):
        pass
    
    def remember(self):
        pass