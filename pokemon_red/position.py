from dataclasses import dataclass
import math
import sys

from .maps import Maps

not_initiated = 1000


@dataclass
class Position:
    def __init__(self, map_: Maps, x: int, y: int):
        self.map = map_
        self.y = y
        self.x = x

    def distance_to(self, other: "Position") -> int:
        if self.map is not other.map:
            return not_initiated
        else:
            return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def __str__(self):
        return f"Map {self.map}, x {self.x}, y {self.y}"


class Positions:
    def __init__(self, map_: Maps, x_start: int, x_end: int, y: int):
        self.positions = list(
            map(lambda x: Position(map_, x, y), range(x_start, x_end + 1))
        )

    def distance_to(self, other: Position) -> int:
        return min(map(lambda position: position.distance_to(other), self.positions))


@dataclass
class YPosition:
    _map: Maps
    y: int

    def intersects_with(self, other: Position):
        return self._map == other.map and self.y == other.y
