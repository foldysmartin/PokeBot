import sys
from time import sleep
from poke_bot import PokemonRedEnv
from sb3_contrib import RecurrentPPO

import keyboard

keep_running = True
manual_control = True


def manual_input():
    while True:
        event = keyboard.read_event()

        if event.event_type == keyboard.KEY_DOWN and event.name == "a":
            return 0

        if event.event_type == keyboard.KEY_DOWN and event.name == "b":
            return 1

        if event.event_type == keyboard.KEY_DOWN and event.name == "left":
            return 2

        if event.event_type == keyboard.KEY_DOWN and event.name == "right":
            return 3

        if event.event_type == keyboard.KEY_DOWN and event.name == "up":
            return 4

        if event.event_type == keyboard.KEY_DOWN and event.name == "down":
            return 5


def run():
    env = PokemonRedEnv(debug=True, step_limit=200000)
    obs, info = env.reset()
    terminated = False
    total_reward = 0
    lstm_states = None

    steps = 0
    # model = RecurrentPPO.load("model/v3")
    while not terminated or keep_running:
        if manual_control:
            action = manual_input()

        # action, lstm_states = model.predict(obs, state=lstm_states, deterministic=False)
        (obs, reward, terminated, done, info) = env.step(action, deterministic=True)
        total_reward += reward
        steps += 1

        if reward:
            print(f"{total_reward} in {steps}")
            steps = 0

        if keyboard.is_pressed("p"):
            print(env.pokemon_red.current_position())

        if keyboard.is_pressed("9"):
            break


run()
