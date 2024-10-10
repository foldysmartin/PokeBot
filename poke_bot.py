from enum import Enum, IntEnum
from genericpath import exists
from math import sqrt
import os
from pathlib import Path

import cv2
import gymnasium as gym
from gymnasium.spaces import Dict, MultiDiscrete, Discrete, Box
import numpy as np
from pyboy import PyBoy

actions = ['a', 'b', 'left', 'right', 'up', 'down', 'start', 'select']
action_space = Discrete(len(actions))

max_row_length = 3,



step_limit = 50

class Maps(IntEnum):
    NotSet = -1
    PalletTown = 0
    Viridian = 1
    Route1 = 12
    PlayerLower = 37
    PlayerUpper = 38
    BLUES_HOUSE = 39
    OaksLab = 40
    ViridianMart = 42

class Position():
    def __init__(self, map, x, y):
        self.map = map
        self.y = y
        self.x = x

class MileStone():
    def __init__(self, condition, name):
        self.condition = condition
        self.name = name

    def reward(self, env):
        if self.condition(env):
            return 1
        else:
            return 0
        
    def __str__(self) -> str:
        return f"{self.name} is {self.complete}"
        
class EventMilestone(MileStone):
    def __init__(self, name, address, bit):
        condition = lambda env: env.read_bit(address, bit)
        super().__init__(condition, name)

class MapMilestone(MileStone):
    def __init__(self, name, map):
        condition = lambda env: env.is_map(map)        
        super().__init__(condition, name)

class PokemonRedEnv(gym.Env):
    def __init__(self, milestones, debug=False, step_limit = 100 ) -> None:
        super(PokemonRedEnv, self).__init__()

        if debug:
            self.pyboy = PyBoy('pokemon_red.gb')
        else:
            self.pyboy = PyBoy('pokemon_red.gb', window="null")
            self.pyboy.set_emulation_speed(0)

        self.milestones = milestones

        self.debug = debug
        self.action_space = action_space

        game_area_observation_space = Dict(
        {
            #"position" : MultiDiscrete([100,100,100]), #Map x y
            "screen": Box(low=0, high=255, shape=self._screen().shape, dtype=np.uint8),
            "events" : MultiDiscrete([2, 2])
        }
        )
        self.observation_space = game_area_observation_space
        self.step_limit = step_limit

        self.reset()

    def reset(self, **kwargs):
        with open(Path('pokemon_red.gb.state'), 'rb') as f:
            self.pyboy.load_state(f)

        
        self.steps = 0  
        self.state_reward = 0

        
        oak_appears = EventMilestone("oak_appears", 0xD74B, 7)        
        follow_oak = EventMilestone("follow_oak", 0xD74B, 0)
        self.events = [oak_appears, follow_oak]

        self.pyboy.tick(20)
        return self._observation(), {}
    
    def step(self, action):
        self.pyboy.button(actions[action], 10)
        self.pyboy.tick(20, self.debug)      
        

        self.steps += 1

        terminate = False    
        state_reward = sum(map(lambda milestone: milestone.reward(self), self.milestones))
        reward = 0
        info = {}

        if state_reward > self.state_reward:
            #print(f"REWARD {state_reward} in {self.steps}")
            reward = state_reward - self.state_reward
            self.state_reward = state_reward

            terminate = True

            info = {
                "sucess": True
            }
        

        if self.steps == self.step_limit:
            terminate = True
            info = {
                "sucess": False
            }           

        return self._observation(), reward, terminate, False, info
    
    def read_bit(self, address, bit):
        return bin(256 + self.pyboy.memory[address])[-bit-1] == '1'

    def is_map(self, expected_map):
        return self._map == expected_map
    
    # def print(self):
    #     data = self.pyboy.screen.ndarray[:,:,0]
    #     cv2.imwrite('output.png', data)
    
    @property
    def _map(self):
        try:
            MAP_N_ADDRESS = 0xD35E
            return Maps(self.pyboy.memory[MAP_N_ADDRESS])
        except:
            print(self.pyboy.memory[MAP_N_ADDRESS])
            return Maps.NotSet

    def _screen(self):
        return self.pyboy.screen.ndarray[:,:,0]  # (144, 160)
    
    def _observation(self):
        _event_status = list(map(lambda milestone: int(milestone.reward(self)), self.events))    

        return {
            "screen": self._screen(),
            "events": _event_status
        }
            
    
    def _coords(self):
        map = self._map
        x = self.pyboy.memory[0xD362]
        y = self.pyboy.memory[0xD361]
        return Position(map, x, y)