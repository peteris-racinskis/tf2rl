import gym
from gym import spaces
import joblib
import numpy as np

class DummyEnv(gym.Env):

    def __init__(self, initial_state):
        super(DummyEnv, self).__init__()
        self.shape = (6,)
        assert initial_state.shape == self.shape
        self.action_space = spaces.Box(self.shape)
        self.observation_space = spaces.Box(self.shape)
        self._max_episodes = 50 # That's about how long the trajectory should god
        self._episode = 0

    # simply return the action as the next observation.
    # reward is something that doesn't matter here eihter.
    # done if we hit max episodes.
    def step(self, action):
        self._episode += 1
        done = (self.episode > self._max_episodes)
        reward = -1
        return np.concatenate([action,self.target_coords], axis=1), reward, done, {}
    
    def reset(self, initial_state, target=3):
        self.initial = initial_state # 
        self.target_coords = initial_state["obses"][-target:]


if __name__ == "__main__":
    data = joblib.load("results/expert/step_00020000_epi_00_return_-0123.3392.pkl")
    env = DummyEnv()
    env.step()
    pass