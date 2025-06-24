from envpong import PongLogic
import random

# Random bot
class BotRight:
    def __init__(self, env):
        self.env = env
        
        # This bot doesn't require an initial observation
        self.obs = None
    
    def act(self):
        action = random.choice([PongLogic.PaddleMove.DOWN, PongLogic.PaddleMove.STILL, PongLogic.PaddleMove.UP])  
        return action
    
    def observe(self, obs):
        self.obs = obs
        
        
# Ball tracking bot
class BotLeft:
    def __init__(self, env):
        self.env = env
        
        # This bot requires an initial observation, set everything to zero
        self.obs = [0]*len(env.observation_space.sample())
    
    def act(self):
        # ball tracking strategy
        p1y = self.obs[1]   # player 1 vertical position
        bally = self.obs[9] # ball vertical position
        
        if p1y < bally:
            action = PongLogic.PaddleMove.UP
        else:
            action = PongLogic.PaddleMove.DOWN
            
        return action
    
    def observe(self, obs):
        self.obs = obs