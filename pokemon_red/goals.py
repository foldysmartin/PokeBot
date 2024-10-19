from pokemon_red.maps import Maps
from pokemon_red.milestone import EventMilestone, MapMilestone, PositionMilestone
from pokemon_red.events import *
from pokemon_red.position import YPosition


def goals():
    return _get_starter() + _get_oaks_parcel()


def _get_starter():
    outside = MapMilestone(Maps.PalletTown)
    downstair = MapMilestone(Maps.PlayerLower)
    oak_appears = EventMilestone(oak_appeared)

    follow_oak = EventMilestone(followed_oak)
    oak_says_pick_starter = EventMilestone(asked_to_choose_starter)
    select_starter = EventMilestone(
        got_starter,
    )
    battle_rival = EventMilestone(battled_rival_lab)

    return [
        downstair,
        outside,
        oak_appears,
        follow_oak,
        oak_says_pick_starter,
        select_starter,
        battle_rival,
    ]


def _get_oaks_parcel():
    outside_with_starter = MapMilestone(Maps.PalletTown, trigger=battled_rival_lab)
    route1 = MapMilestone(Maps.Route1)
    route1_check_point_1 = PositionMilestone(YPosition(Maps.Route1, 27))
    potion_sample = EventMilestone(got_potion_sample)
    route1_check_point_2 = PositionMilestone(YPosition(Maps.Route1, 23))
    route1_check_point_3 = PositionMilestone(YPosition(Maps.Route1, 19))
    route1_check_point_4 = PositionMilestone(YPosition(Maps.Route1, 13))
    route1_check_point_5 = PositionMilestone(YPosition(Maps.Route1, 5))
    viridian = MapMilestone(Maps.Viridian)
    enter_viridian_checkpoint = PositionMilestone(YPosition(Maps.Viridian, 27))
    viridian_mart = MapMilestone(Maps.ViridianMart)
    get_oaks_parcel = EventMilestone(got_oaks_parcel)

    return [
        outside_with_starter,
        route1,
        route1_check_point_1,
        potion_sample,
        route1_check_point_2,
        route1_check_point_3,
        route1_check_point_4,
        route1_check_point_5,
        viridian,
        enter_viridian_checkpoint,
        viridian_mart,
        get_oaks_parcel,
    ]
