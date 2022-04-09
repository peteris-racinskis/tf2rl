import gym
from gym import spaces
import joblib
import numpy as np

class DummyEnv(gym.Env):

    def __init__(self):
        super(DummyEnv, self).__init__()
        self.shape = (11,)
        self.action_shape=(8,)
        self.action_space = spaces.Box(-10,10,self.action_shape)
        self.observation_space = spaces.Box(-10,10,self.shape)
        self._max_episodes = 50 # That's about how long the trajectory should go

    # simply return the action as the next observation.
    # reward is something that doesn't matter here eihter.
    # done if we hit max episodes.
    def step(self, action: np.ndarray):
        self._episode += 1
        done = (self._episode >= self._max_episodes)
        reward = -1
        return np.concatenate([action.flatten(),self.target_coords]), reward, done, {}
    
    def reset(self, initial_state, target=3):
        assert initial_state.shape[-1:] == self.shape
        self._episode = 0
        self.initial = initial_state # 
        self.target_coords = initial_state.flatten()[-target:]
        return initial_state


if __name__ == "__main__":
    data = joblib.load("results/expert/step_00020000_epi_00_return_-0123.3392.pkl")
    env = DummyEnv()
    env.step()
    pass