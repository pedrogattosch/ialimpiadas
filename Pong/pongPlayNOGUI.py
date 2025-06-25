from envpong import PongEnv, PongLogic
from bot import BotLeft 
import os
import random

def runLoop(env):
    tracker_bot = BotLeft(env, is_training=False) 

    obs, info = env.reset()

    for i in range(5000):
        actionp1 = tracker_bot.act(obs)
        
        actionp2 = random.choice([PongLogic.PaddleMove.DOWN, PongLogic.PaddleMove.STILL, PongLogic.PaddleMove.UP])
         
        new_obs, reward, done, truncated, info = env.step(actionp1, actionp2)
        
        obs = new_obs 

        if done:
            obs, info = env.reset()

def main():
    env = PongEnv(debugPrint=True)
    runLoop(env)
        

if __name__ == "__main__":
    file_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(file_path)

    main()