from dataclasses import dataclass


@dataclass
class Event:
    address: int
    bit: int


oak_appeared = Event(0xD74B, 7)
followed_oak = Event(0xD74B, 0)
asked_to_choose_starter = Event(0xD74B, 1)
got_starter = Event(0xD74B, 2)
battled_rival_lab = Event(0xD74B, 3)
got_potion_sample = Event(0xD7BF, 0)
got_oaks_parcel = Event(0xD74E, 1)
