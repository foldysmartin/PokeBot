import sys
from time import sleep

from stable_baselines3 import PPO
from poke_bot_encoded import PokemonRedEnv

import keyboard

keep_running = False
manual_control = False


def run():
    env = PokemonRedEnv(
        debug=True,
        step_limit=20000,
        manual_input=manual_control,
    )
    obs, info = env.reset()
    terminated = False
    total_reward = 0
    model = PPO.load("oaks_parcel")
    while not terminated or keep_running:
        action, _states = model.predict(obs, deterministic=False)
        # action = 1
        (obs, reward, terminated, done, info) = env.step(action)
        total_reward += reward

        if reward:
            print(total_reward)

        if keyboard.is_pressed("p"):
            print(env.pokemon_red.current_position())

        if keyboard.is_pressed("9"):
            break


run()
