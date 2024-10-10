from dataclasses import dataclass
from decimal import Decimal
from genericpath import exists
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import uuid
import math

import warnings

from poke_bot_encoded import (
    PokemonRedEnv,
    step_limit,
    MapMilestone,
    EventMilestone,
    Maps,
    MileStone,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)

from sb3_contrib import RecurrentPPO

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common import env_checker
from stable_baselines3.common.utils import constant_fn
from stable_baselines3.common.evaluation import evaluate_policy

session_path = "Sessions/"
tensorboard_path = "Tensorboard/"


def train():
    sess_path = Path(f"{session_path}/session_{str(uuid.uuid4())[:8]}")
    ep_length = 10000

    env = DummyVecEnv([lambda: Monitor(PokemonRedEnv(False))])
    checkpoint_callback = CheckpointCallback(
        save_freq=ep_length,
        save_path=sess_path,
        save_replay_buffer=True,
        save_vecnormalize=True,
        name_prefix="train",
    )

    learn_steps = 40

    file_name = f"Sessions/session_6d84654c/train_491520_steps"

    if exists(file_name + ".zip"):
        print("\nloading checkpoint")
        model = PPO.load(file_name, env=env, tensorboard_log=tensorboard_path)
        model.n_steps = ep_length
        model.n_envs = 1
        model.rollout_buffer.buffer_size = ep_length
        model.rollout_buffer.n_envs = 1
        model.rollout_buffer.reset()
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            n_steps=ep_length,
            batch_size=ep_length,
            n_epochs=1,
            tensorboard_log=tensorboard_path,
            gamma=0.99,
        )

    for i in range(learn_steps):
        model.learn(
            total_timesteps=ep_length * 100,
            callback=checkpoint_callback,
            tb_log_name="first_run",
            reset_num_timesteps=True,
            progress_bar=True,
        )
        episode_rewards, episode_lengths = evaluate_policy(
            model,
            model.get_env(),
            n_eval_episodes=1,
            deterministic=False,
            return_episode_rewards=True,
        )
        episode_rewards, episode_lengths = evaluate_policy(
            model,
            model.get_env(),
            n_eval_episodes=10,
            deterministic=False,
            return_episode_rewards=True,
        )
        model.save(f"model/v{i}")
        with open("results.txt", "w") as file:
            # Write a string to the file
            text = (
                f"Nondeterministic Result reward = {episode_rewards}, steps = {episode_lengths}"
                + "\n"
                + f"Deterministic Result reward = {episode_rewards}, steps = {episode_lengths}"
            )
            file.write(text)


@dataclass
class MilestoneHolder:
    milestone: MileStone
    steps: int

    def gamma(self):
        return 0.1 ** (1 / self.steps)


def train_v2_train(milestones):
    sess_path = Path(f"{session_path}/session_{str(uuid.uuid4())[:8]}")
    ep_length = 2048 * 8

    env = DummyVecEnv(
        [
            lambda: Monitor(
                PokemonRedEnv(debug=False, step_limit=ep_length, milestones=[])
            )
        ]
    )

    learn_steps = 40

    file_name = f"Sessions/session_6d84654c/train_491520_steps"

    if exists(file_name + ".zip"):
        print("\nloading checkpoint")
        model = PPO.load(file_name, env=env, tensorboard_log=tensorboard_path)
        model.n_steps = ep_length
        model.n_envs = 1
        model.rollout_buffer.buffer_size = ep_length
        model.rollout_buffer.n_envs = 1
        model.rollout_buffer.reset()
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            n_steps=ep_length,
            batch_size=512,
            n_epochs=1,
            tensorboard_log=tensorboard_path,
        )

    for milestone in milestones:
        evaluated = False
        while not evaluated:
            model.gamma = milestone.gamma()
            ep_length = milestone.steps * 10
            model.env = DummyVecEnv(
                [
                    lambda: Monitor(
                        PokemonRedEnv(
                            debug=False,
                            step_limit=ep_length,
                            milestones=[milestone.milestone],
                        )
                    )
                ]
            )

            checkpoint_callback = CheckpointCallback(
                save_freq=ep_length,
                save_path=sess_path,
                save_replay_buffer=True,
                save_vecnormalize=True,
                name_prefix=milestone.milestone.name,
            )
            model.learn(
                total_timesteps=ep_length * 100,
                callback=checkpoint_callback,
                tb_log_name=milestone.milestone.name,
                reset_num_timesteps=True,
                progress_bar=True,
            )

            episode_rewards, episode_lengths = evaluate_policy(
                model,
                model.get_env(),
                n_eval_episodes=10,
                deterministic=False,
                return_episode_rewards=True,
            )

            # all rewards 1 and length is acceptable
            if all(ele == 1 for ele in episode_rewards) and all(
                ele <= milestone.steps * 2 for ele in episode_lengths
            ):
                evaluated = True

            print(f"Result reward = {episode_rewards}, steps = {episode_lengths}")
            model.save(f"model/{milestone.milestone.name}.zip")


def train_v2():
    outside = MapMilestone("outside", Maps.PalletTown)
    downstair = MapMilestone("downstairs", Maps.PlayerLower)
    oak_appears = EventMilestone("oak_appears", 0xD74B, 7)
    milestones = [
        MilestoneHolder(downstair, 9),
        MilestoneHolder(outside, 11),
        MilestoneHolder(oak_appears, 10),
    ]

    total_steps = 0

    for milestone in milestones:
        total_steps += milestone.steps
        milestone.steps = total_steps

    train_v2_train(milestones)


if __name__ == "__main__":
    train()
