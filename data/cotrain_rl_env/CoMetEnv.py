import gym
from gym import spaces
import pandas as pd
import numpy as np

class CoMetEnv(gym.Env):
    """Custom Environment that follows gym interface
        metadata = {'render.modes': ['human']}

    def __init__(self, arg1, arg2, ...):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=255, shape=
        (HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)

        ...


    """

    def __init__(self, arg1):
        pass

    def step(self, action):
        # Execute one time step within the environment
        pass

    def reset(self):
        pass
