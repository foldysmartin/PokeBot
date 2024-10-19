from pyboy import PyBoy
from pathlib import Path

from .maps import Maps
from .events import Event
from .position import Position

actions = ["a", "b", "left", "right", "up", "down", "start", "select"]


class PokemonRed:
    def __init__(self, debug=False, manual_input=False, path="") -> None:
        if debug:
            self._pyboy = PyBoy("pokemon_red.gb")
        else:
            self._pyboy = PyBoy("pokemon_red.gb", window="null")
            self._pyboy.set_emulation_speed(0)

        self._debug = debug
        self._manual_input = manual_input
        self.path = path

    def input(self, action):
        if not self._manual_input:
            self._pyboy.button(action, 10)

        self._pyboy.tick(20, self._debug)

    def reset(self):
        path = Path(self.path + "pokemon_red.gb.state")
        with open(path, "rb") as f:
            self._pyboy.load_state(f)

        self._pyboy.tick(20)

    def screen(self):
        return self._pyboy.screen.ndarray[:, :, 0:1]  # (144, 160)

    def current_position(self):
        map = self._map
        x = self._read_memory(0xD362)
        y = self._read_memory(0xD361)
        return Position(map, x, y)

    def event_complete(self, event: Event):
        return bin(256 + self._read_memory(event.address))[-event.bit - 1] == "1"

    def save_state(self, file_name):
        with open(file_name, "r+"):
            self._pyboy.save_state(file_name)

    @property
    def _map(self):
        try:
            MAP_N_ADDRESS = 0xD35E
            return Maps(self._read_memory(MAP_N_ADDRESS))
        except:
            print(self._read_memory(MAP_N_ADDRESS))
            return Maps.NotSet

    def _read_memory(self, address):
        return self._pyboy.memory[address]
