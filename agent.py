#Import retro to play Sonic The Hedgehog 2
import retro
#Import time to slow down the game
import time
#Import for preprocessing
#Import environment base class for a wrapper
from gym import Env
#Import the space shapes for the environment
from gym.spaces import MultiBinary, Box
#Import numpy to calculate frame delta
import numpy as np
#Import opencv for grayscaling
import cv2
#Import matplotlib for plotting the image
from matplotlib import pyplot as plt

import math
#import torch



#retro.data.list_games()

#Create custom environment
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

def _seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**32
        return [seed1, seed2] 



# Starts de game environment - It can only be running one at a time
env = SonicTheHedgehog2()
env.seed(0)

print("The size of frame is: ", env.observation_space.shape)
print("No. of Actions: ", env.action_space.n)


#Limit Posible Actions
posible_actions = {
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



# Sample the observation space
#print(env.observation_space)


# Reset game to starting state
obs = env.reset()

# Grayscaling
#gray = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
# Resize
#resize = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_CUBIC)
# Plot
#plt.imshow(cv2.cvtColor(resize, cv2.COLOR_BGR2RGB))

# Set flag to false
done = False

# We will play only one game
print(retro.Actions.FILTERED)
for game in range(1):
    while not done:
        if done:
            obs = env.reset()
        env.render()
        #Takes random desicions
        obs, reward, done, info = env.step(env.action_space.sample())
        #time.sleep(0.005)
        
        if reward > 0:
            print(reward)
        
    print(info)


#Close Previous Environment if exists
env.close()
