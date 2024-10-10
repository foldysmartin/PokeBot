from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from poke_bot_encoded import MapMilestone, Maps, PokemonRedEnv

from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.evaluation import evaluate_policy
import keyboard
from pyboy.api.memory_scanner import DynamicComparisonType


def run():

    # downstair = MapMilestone("downsairs", Maps.PlayerLower)
    env = PokemonRedEnv(debug=True)
    model = PPO.load("model/v6")
    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=False, return_episode_rewards=True)
    # print(mean_reward)
    obs, info = env.reset()
    terminated = False

    memory = True
    while not terminated:
        action, _ = model.predict(obs, deterministic=False)
        (
            obs,
            rewards,
            terminated,
            done,
            info,
        ) = env.step(action)

        if keyboard.is_pressed("1"):
            print("Refresh")
            memory = env.pyboy.memory_scanner.scan_memory()
            print(len(memory))

        if keyboard.is_pressed("2"):
            print("Unchanged")
            memory = env.pyboy.memory_scanner.rescan_memory(
                dynamic_comparison_type=DynamicComparisonType.UNCHANGED
            )
            print(len(memory))

        if keyboard.is_pressed("3"):
            print("Changed")
            memory = env.pyboy.memory_scanner.rescan_memory(
                dynamic_comparison_type=DynamicComparisonType.CHANGED
            )
            print(len(memory))

        if keyboard.is_pressed("4"):
            a = [
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
            print(list(map(lambda mem: env.pyboy.memory[mem], a)))

        if keyboard.is_pressed("5"):
            # print(f"X position {env.pyboy.memory[0xCC24]}")
            # print(f"Y position {env.pyboy.memory[0xCC25]}")
            # print(f"current position {env.pyboy.memory[0xCC26]}")
            # print(f"party position {env.pyboy.memory[0xCC2B]}")
            # print(f"iten position {env.pyboy.memory[0xCC2C]}")
            # print(f"start position {env.pyboy.memory[0xCC2D]}")

            print("Pokemon 1 ")

        if keyboard.is_pressed("m"):
            break


if __name__ == "__main__":
    run()
