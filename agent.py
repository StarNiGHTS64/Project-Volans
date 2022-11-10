#Import retro to play Sonic The Hedgehog 2
import retro
#Import time to slow down the game
import time

retro.data.list_games()

def _seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**32
        return [seed1, seed2] 

# Starts de game environment - It can only be running one at a time
env = retro.make(game='SonicTheHedgehog2-Genesis', state='ChemicalPlantZone.Act1', scenario='contest')
#env.seed(0)
# Sample the observation space
print(env.observation_space)


# Reset game to starting state
obs = env.reset()

# Set flag to false
done = False

# We will play only one game
for game in range(1):
    while not done:
        if done:
            obs = env.reset()
        env.render()
        #Takes random desicions
        obs, reward, done, info = env.step(env.action_space.sample())
        #time.sleep(0.005)
        print(reward)
        #print(info)


#Close Previous Environment if exists
env.close()

