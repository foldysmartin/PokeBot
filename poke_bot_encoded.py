import gymnasium as gym
from gymnasium.spaces import Dict, MultiDiscrete, Discrete
from pokemon_red.goals import goals
from pokemon_red.milestone import EventMilestone
from pokemon_red.pokemon_red import PokemonRed, actions
from pokemon_red.events import *

action_space = Discrete(len(actions))

statistics = []


class PokemonRedEnv(gym.Env):
    def __init__(
        self, debug=False, step_limit=10000, manual_input=False, path=""
    ) -> None:
        super(PokemonRedEnv, self).__init__()
        self.pokemon_red = PokemonRed(debug, manual_input, path)
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

        _events = list(
            filter(
                lambda milestone: isinstance(milestone, EventMilestone),
                goals(),
            )
        )

        game_area = {
            "position": MultiDiscrete([100, 100, 100]),  # Map x y
            "events": MultiDiscrete([2] * len(_events)),
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

        game_area_observation_space = Dict(
            {**game_area, **pokemon("party", 1), **battle}
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
        info = {}
        self.total += reward

        if reward > 0:
            print(f"Reward {self.total} in {self.steps}")

        if self.milestones[-1].complete:
            terminate = True

        if self.steps >= self.step_limit:
            terminate = True

        return self._observation(), reward, terminate, False, info

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
        return list(
            map(lambda address: self.pokemon_red._read_memory(address), addresses)
        )

    def _pokemon_observation(self, location, slot):
        return {
            f"id_{location}_{slot}": self.pokemon_red._read_memory(0xD16B),
            f"current_hp_{location}_{slot}": self._read_two(0xD16C),
            f"move1_{location}_{slot}": self.pokemon_red._read_memory(0xD173),
            f"pp1_{location}_{slot}": self.pokemon_red._read_memory(0xD188),
            f"move2_{location}_{slot}": self.pokemon_red._read_memory(0xD174),
            f"pp2_{location}_{slot}": self.pokemon_red._read_memory(0xD189),
            f"move3_{location}_{slot}": self.pokemon_red._read_memory(0xD175),
            f"pp3_{location}_{slot}": self.pokemon_red._read_memory(0xD18A),
            f"move4_{location}_{slot}": self.pokemon_red._read_memory(0xD176),
            f"pp4_{location}_{slot}": self.pokemon_red._read_memory(0xD18B),
            f"level_{location}_{slot}": self.pokemon_red._read_memory(0xD18C),
            f"hp_{location}_{slot}": self._read_two(0xD18D),
            f"attack_{location}_{slot}": self._read_two(0xD18F),
            f"defence_{location}_{slot}": self._read_two(0xD191),
            f"speed_{location}_{slot}": self._read_two(0xD193),
            f"special_{location}_{slot}": self._read_two(0xD195),
        }

    def _observation(self):
        _coords = self.pokemon_red.current_position()
        _events = filter(
            lambda milestone: isinstance(milestone, EventMilestone), goals()
        )
        _event_status = list(map(lambda milestone: int(milestone.complete), _events))

        _game_area = {
            "position": [int(_coords.map), _coords.x, _coords.y],
            "events": _event_status,
            "text_box": self._txt_box(),
            "cursor": [
                self.pokemon_red._read_memory(0xCC24),
                self.pokemon_red._read_memory(0xCC25),
                self.pokemon_red._read_memory(0xCC26),
                self.pokemon_red._read_memory(0xCC2B),
                self.pokemon_red._read_memory(0xCC2C),
                self.pokemon_red._read_memory(0xCC2D),
            ],
        }
        _pokemon = self._pokemon_observation("party", 1)
        _battle = {
            "own_id": self.pokemon_red._read_memory(0xD014),
            "own_hp": self._read_two(0x015),
            "enemy_id": self.pokemon_red._read_memory(0xCFE5),
            "enemy_hp": self._read_two(0xCFE6),
        }

        observation = {**_game_area, **_pokemon, **_battle}

        import json

        with open("data.json", "w") as json_file:
            json.dump(observation, json_file)

        return observation

    def _read_two(self, start):
        return 256 * self.pokemon_red._read_memory(
            start
        ) + self.pokemon_red._read_memory(start + 1)
