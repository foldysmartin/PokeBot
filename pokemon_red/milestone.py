from dataclasses import dataclass
import sys

from pokemon_red.pokemon_red import PokemonRed

from .position import YPosition
from .events import Event
from .maps import Maps


@dataclass
class EventMilestone:
    event: Event
    complete: bool = False

    def reward(self, game: PokemonRed):
        if self.complete:
            return 0

        if game.event_complete(self.event):
            self.complete = True
            return 1
        else:
            return 0


@dataclass
class PositionMilestone:
    target: YPosition

    trigger: Event = None

    complete: bool = False

    def reward(self, game: PokemonRed):
        if self.complete:
            return 0

        if self.trigger and not game.event_complete(self.trigger):
            return 0

        if self.target.intersects_with(game.current_position()):
            self.complete = True
            return 1

        return 0


@dataclass
class MapMilestone:
    map: Maps
    trigger: Event = None
    complete: bool = False

    def reward(self, game: PokemonRed):
        if self.complete:
            return 0

        if self.trigger and not game.event_complete(self.trigger):

            return 0

        if self.map == game.current_position().map:
            self.complete = True
            return 1
        return 0

    # def reward(self, game: PokemonRed):
    #     if self.complete:
    #         return 0

    #     if self.trigger and not game.event_complete(self.trigger):
    #         return 0

    #     _distance_to_target = self.target.distance_to(game.current_position())

    #     if self._is_complete(game):
    #         self.complete = True
    #         return 1
    #     elif _distance_to_target < self._previous_best_distance_to_target:
    #         reward = int(self._previous_best_distance_to_target != not_initiated)
    #         self._previous_best_distance_to_target = _distance_to_target
    #         return reward
    #     else:
    #         return 0

    # def _is_complete(self, game: PokemonRed):
    #     if type(self.compete_check) is Maps:
    #         return game.current_position().map == self.compete_check
    #     else:
    #         return game.event_complete(self.compete_check)
