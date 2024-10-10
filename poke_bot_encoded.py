from enum import Enum, IntEnum
from genericpath import exists
from math import sqrt
import os
from pathlib import Path

import gymnasium as gym
from gymnasium.spaces import Dict, MultiDiscrete, Discrete, Box
import numpy as np
from pyboy import PyBoy

actions = ["a", "b", "left", "right", "up", "down", "start", "select"]
action_space = Discrete(len(actions))

max_row_length = (3,)


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


class Position:
    def __init__(self, map, x, y):
        self.map = map
        self.y = y
        self.x = x


class MileStone:
    def __init__(self, condition, name):
        self.condition = condition
        self.name = name
        self.complete = False

    def reward(self, env):
        if self.condition(env) or self.complete:
            self.complete = True
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
    def __init__(self, debug=False, step_limit=10000) -> None:
        super(PokemonRedEnv, self).__init__()

        if debug:
            self.pyboy = PyBoy("pokemon_red.gb")
        else:
            self.pyboy = PyBoy("pokemon_red.gb", window="null")
            self.pyboy.set_emulation_speed(0)

        self.debug = debug
        self.action_space = action_space

        def pokemon(location, slot):
            return {
                f"id_{location}_{slot}": Discrete(191),
                f"current_hp_{location}_{slot}": Discrete(704),
                f"move1_{location}_{slot}": Discrete(166),
                f"pp1_{location}_{slot}": Discrete(62),
                f"move2_{location}_{slot}": Discrete(166),
                f"pp2_{location}_{slot}": Discrete(62),
                f"move3_{location}_{slot}": Discrete(166),
                f"pp3_{location}_{slot}": Discrete(62),
                f"move4_{location}_{slot}": Discrete(166),
                f"pp4_{location}_{slot}": Discrete(62),
                f"level_{location}_{slot}": Discrete(101),
                f"hp_{location}_{slot}": Discrete(704),
                f"attack_{location}_{slot}": Discrete(367),
                f"defence_{location}_{slot}": Discrete(459),
                f"speed_{location}_{slot}": Discrete(407),
                f"special_{location}_{slot}": Discrete(379),
            }

        game_area = {
            "position": MultiDiscrete([100, 100, 100]),  # Map x y
            "events": MultiDiscrete([2] * 5),
            "text_box": MultiDiscrete(
                [256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256]
            ),
            "cursor": MultiDiscrete(
                [20, 20, 10, 10, 10, 10]
            ),  # X, Y, current, party, item, start
        }

        battle = {
            "own_id": Discrete(191),
            "own_hp": Discrete(704),
            "enemy_id": Discrete(191),
            "enemy_hp": Discrete(704),
        }

        game_area_observation_space = Dict({**game_area, **pokemon("party", 1)})
        self.observation_space = game_area_observation_space
        self.step_limit = step_limit

        print("init")
        self.reset()

    def reset(self, **kwargs):
        print("reset")
        with open(Path("pokemon_red.gb.state"), "rb") as f:
            self.pyboy.load_state(f)

        outside = MapMilestone("outside", Maps.PalletTown)
        downstair = MapMilestone("downstairs", Maps.PlayerLower)
        oak_appears = EventMilestone("oak_appears", 0xD74B, 7)

        follow_oak = EventMilestone("follow_oak", 0xD74B, 0)
        asked_to_choose_mon = EventMilestone("asked_to_choose_mon", 0xD74B, 1)
        got_starter = EventMilestone("got_starter", 0xD74B, 2)
        battled_rival_lab = EventMilestone("battled_rival_lab", 0xD74B, 3)

        self.milestones = [
            downstair,
            outside,
            oak_appears,
            follow_oak,
            asked_to_choose_mon,
            got_starter,
            battled_rival_lab,
        ]
        self.steps = 0
        self.state_reward = 0

        self.pyboy.tick(20)
        return self._observation(), {}

    def step(self, action):
        self.pyboy.button(actions[action], 10)
        self.pyboy.tick(20, self.debug)

        self.steps += 1

        terminate = False
        state_reward = sum(
            map(lambda milestone: milestone.reward(self), self.milestones)
        )
        reward = 0
        info = {}

        if state_reward > self.state_reward:
            print(f"REWARD {state_reward} in {self.steps}")
            reward = state_reward - self.state_reward
            self.state_reward = state_reward

        if self.steps == self.step_limit:
            terminate = True

            for milestone in self.milestones:
                print(milestone.__str__())
            info = {"sucess": False}

        if self.milestones[-1].complete:
            terminate = True

            for milestone in self.milestones:
                print(milestone.__str__())

            info = {"sucess": True}

        return self._observation(), reward, terminate, False, info

    def read_bit(self, address, bit):
        return bin(256 + self.pyboy.memory[address])[-bit - 1] == "1"

    def is_map(self, expected_map):
        return self._map == expected_map

    @property
    def _map(self):
        try:
            MAP_N_ADDRESS = 0xD35E
            return Maps(self.pyboy.memory[MAP_N_ADDRESS])
        except:
            print(self.pyboy.memory[MAP_N_ADDRESS])
            return Maps.NotSet

    def _screen(self):
        return self.pyboy.screen.ndarray[:, :, 0]  # (144, 160)

    def _txt_box(self):
        addresses = [
            50406,
            50407,
            50408,
            50409,
            52308,
            53010,
            58598,
            58599,
            58600,
            58601,
            60500,
            61202,
            65493,
        ]
        return list(map(lambda address: self.pyboy.memory[address], addresses))

    def _pokemon_observation(self, location, slot):
        return {
            f"id_{location}_{slot}": self.pyboy.memory[0xD16B],
            f"current_hp_{location}_{slot}": self._read_two(0xD16C),
            f"move1_{location}_{slot}": self.pyboy.memory[0xD173],
            f"pp1_{location}_{slot}": self.pyboy.memory[0xD188],
            f"move2_{location}_{slot}": self.pyboy.memory[0xD174],
            f"pp2_{location}_{slot}": self.pyboy.memory[0xD189],
            f"move3_{location}_{slot}": self.pyboy.memory[0xD175],
            f"pp3_{location}_{slot}": self.pyboy.memory[0xD18A],
            f"move4_{location}_{slot}": self.pyboy.memory[0xD176],
            f"pp4_{location}_{slot}": self.pyboy.memory[0xD18B],
            f"level_{location}_{slot}": self.pyboy.memory[0xD18C],
            f"hp_{location}_{slot}": self._read_two(0xD18D),
            f"attack_{location}_{slot}": self._read_two(0xD18F),
            f"defence_{location}_{slot}": self._read_two(0xD191),
            f"speed_{location}_{slot}": self._read_two(0xD193),
            f"special_{location}_{slot}": self._read_two(0xD195),
        }

    def _observation(self):
        _coords = self._coords()
        _events = filter(
            lambda milestone: isinstance(milestone, EventMilestone), self.milestones
        )
        _event_status = list(map(lambda milestone: int(milestone.complete), _events))

        _game_area = {
            "position": [int(_coords.map), _coords.x, _coords.y],
            "events": _event_status,
            "text_box": self._txt_box(),
            "cursor": [
                self.pyboy.memory[0xCC24],
                self.pyboy.memory[0xCC25],
                self.pyboy.memory[0xCC26],
                self.pyboy.memory[0xCC2B],
                self.pyboy.memory[0xCC2C],
                self.pyboy.memory[0xCC2D],
            ],
        }
        _pokemon = self._pokemon_observation("party", 1)
        _battle = {
            "own_id": self.pyboy.memory[0xD014],
            "own_hp": self._read_two(0x015),
            "enemy_id": self.pyboy.memory[0xCFD8],
            "enemy_hp": self._read_two(0xCFE6),
        }

        import json

        with open("data.json", "w") as json_file:
            json.dump({**_game_area, **_pokemon, **_battle}, json_file)

        return {**_game_area, **_pokemon}

    def _read_two(self, start):
        return 256 * self.pyboy.memory[start] + self.pyboy.memory[start + 1]

    def _coords(self):
        map = self._map
        x = self.pyboy.memory[0xD362]
        y = self.pyboy.memory[0xD361]
        return Position(map, x, y)
