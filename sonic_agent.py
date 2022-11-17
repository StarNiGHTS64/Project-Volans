import retro
import time
from gym import Env
from gym.spaces import MultiBinary, Box
import numpy as np
import cv2
from matplotlib import pyplot as plt
import torch



import math

#matplotlib inline

# Importing the optimzation frame - HPO
import optuna
# PPO algo for RL
from stable_baselines3 import PPO
# Bring in the eval policy method for metric calculation
from stable_baselines3.common.evaluation import evaluate_policy
# Import the sb3 monitor for logging 
from stable_baselines3.common.monitor import Monitor
# Import the vec wrappers to vectorize and frame stack
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
# Import os to deal with filepaths
import os


class SonicTheHedgehog2(Env):
    def __init__(self):
        super().__init__()
        #Specify action space and observation space
        self.observation_space = Box(low= 0, 
                                    high=255, 
                                    shape=(84, 84, 1), 
                                    dtype=np.uint8)
        self.action_space = MultiBinary(12)
        # Startup an instance of the game
        self.game = retro.make(game='SonicTheHedgehog2-Genesis', 
                                state='EmeraldHillZone.Act1', 
                                scenario='contest',
                                use_restricted_actions=retro.Actions.FILTERED)

    def reset(self):
        # Return the first frame
        obs = self.game.reset()
        #Current Frame - Previous Frame
        obs = self.preprocess(obs)
        self.previous_frame = obs

        # Create a placeholder attribute to hold the score delta
        self.x = 0
        return obs
    
    def preprocess(self, observation):
        # Grayscaling
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        # Resize
        resize = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_CUBIC)
        # Add the channels value
        channels = np.reshape(resize, (84, 84, 1))
        return channels


    def step(self, action):
        # Take a step
        # Recieve the unprocessed items
        obs, reward, done, info = self.game.step(action)
        obs = self.preprocess(obs)

        # Frame Delta
        # Substract previous frame from the Current frame in order to see pixel changes
        frame_delta = obs - self.previous_frame
        self.previous_frame = obs

        #Reshape the reward function TODO
        reward = info['x'] - self.x
        self.x = info['x']

        return frame_delta, reward, done, info


    def render(self, *args, **kwargs):
        self.game.render()

    def close(self):
        self.game.close()

env = SonicTheHedgehog2()
print("The size of frame is: ", env.observation_space.shape)
print("No. of Actions: ", env.action_space.n)

obs = env.reset()

done = False

#Limit Posible Actions
possible_actions = {
    # No Operation
    0: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # Left
    1: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    # Right
    2: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    # Left, Down
    3: [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    # Right, Down
    4: [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    # Down
    5: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    # Down, B
    6: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    # B
    7: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}

"""
plt.figure()
plt.imshow(env.reset())
plt.title('Original Frame')
plt.show

plt.figure()
plt.imshow(cv2.cvtColor(obs, cv2.COLOR_BGR2RGB))
plt.title('Pre Processed image')
plt.show()
"""

score = 0
for game in range(1):
    while not done:
        env.render()
        action = possible_actions[np.random.randint(len(possible_actions))]
        #Takes random desicions
        obs, reward, done, info = env.step(action)
        #time.sleep(0.005)
        score += reward

        if reward > 0:
            print(reward)
            
        if done:
            print("Your Score at the end of the game is: ")
            break

    env.reset()
    env.render(close=True)
    env.close()

LOG_DIR = './logs/'
OPT_DIR = './opt/'