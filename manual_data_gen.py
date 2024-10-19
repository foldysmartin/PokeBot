from random import randint
from poke_bot import PokemonRedEnv, actions
from pokemon_red.milestone import *
from pokemon_red.events import *
from imitation.data.types import Trajectory
from imitation.data import serialize

tmp_state_name = "tmp.gb.state"

targets = [
    (MapMilestone(Maps.PalletTown), 9, "Pallet"),
    (MapMilestone(Maps.PlayerLower), 14, "PlayerLower"),
    (EventMilestone(oak_appeared), 13, "oak_appeared"),
    (EventMilestone(followed_oak), 66, "followed_oak"),
    (EventMilestone(asked_to_choose_starter), 58, "asked_to_choose_starter"),
    (EventMilestone(got_starter), 61, "got_starter"),
    (EventMilestone(battled_rival_lab), 335, "battled_rival_lab"),
]

trajectories = []

for i in range(2):
    env = PokemonRedEnv()
    (obs, info) = env.reset()

    sucessful_actions = []
    observations = [obs]
    load_state = "pokemon_red.gb.state"

    for target in targets:
        print(target[2])
        completed = False
        while not completed:
            (obs, info) = env.reset(
                milestones=[target[0]], step_limit=target[1] * 10, state_file=load_state
            )
            tmp_obs = []
            tmp_actions = []

            terminated = False
            while not terminated:
                action = randint(0, len(actions) - 1)

                (obs, reward, terminated, done, info) = env.step(action)
                tmp_obs.append(obs)
                tmp_actions.append(action)

            if info["successful"]:
                observations.append(tmp_obs)
                sucessful_actions.append(tmp_actions)

                env.pokemon_red.save_state(tmp_state_name)
                load_state = tmp_state_name
                completed = True

    trajectories.append(Trajectory(observations, sucessful_actions))

serialize.save("manual_data", trajectories)
