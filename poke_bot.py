from pokemon_red.goals import goals
from pokemon_red.pokemon_red import PokemonRed, actions
from pokemon_red.events import *


import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np


action_space = Discrete(len(actions))


class PokemonRedEnv(gym.Env):
    def __init__(
        self, debug=False, step_limit=10000, manual_input=False, path=""
    ) -> None:
        super(PokemonRedEnv, self).__init__()
        self.pokemon_red = PokemonRed(debug, manual_input, path)
        self.action_space = action_space

        game_area_observation_space = Box(
            low=0,
            high=255,
            shape=(144, 160, 1),
            dtype=np.uint8,
        )
        self.observation_space = game_area_observation_space
        self.step_limit = step_limit
        self.reset()

    def reset(self, **kwargs):
        self.milestones = goals()
        self.steps = 0
        self.pokemon_red.reset()
        self.total = 0

        return self._observation(), {}

    def step(self, action):
        self.pokemon_red.input(actions[action])
        self.steps += 1

        terminate = False
        reward = sum(
            map(
                lambda milestone: milestone.reward(self.pokemon_red),
                self.milestones,
            )
        )
        info = {"successful": False}

        self.total += reward

        if reward > 0:
            print(f"Reward {self.total} in {self.steps}")

        if self.milestones[-1].complete:
            terminate = True
            info = {"successful": True}

        if self.steps >= self.step_limit:
            terminate = True

        return self._observation(), reward, terminate, False, info

    def read_bit(self, address, bit):
        return bin(256 + self.pyboy.memory[address])[-bit - 1] == "1"

    def _observation(self):
        return self.pokemon_red.screen()
