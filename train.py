from poke_bot import (
    PokemonRedEnv,
)

from genericpath import exists
from pathlib import Path
from stable_baselines3.common.monitor import Monitor

from sb3_contrib import RecurrentPPO

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

drive = "drive/MyDrive/Pokemon/"
goal = "oaks_parcel"
target = 0
session_path = drive + "Sessions/"
tensorboard_path = "Tensorboard/"
ep_length = 40000


def _create_env():
    def func():
        return Monitor(PokemonRedEnv(False, step_limit=ep_length, path=drive))

    return func


def _environments(count):
    return list(map(lambda _: _create_env(), range(count)))


def train():
    sess_path = Path(f"{session_path}/{goal}")
    environment_count = 1
    env = DummyVecEnv(_environments(environment_count))
    checkpoint_callback = CheckpointCallback(
        save_freq=ep_length * 10,
        save_path=sess_path,
        save_replay_buffer=True,
        save_vecnormalize=True,
        name_prefix="train",
    )

    batch_size = ep_length // 10
    file_name = f"{session_path}/oaks_parcel/train_50000_steps"

    if exists(file_name + ".zip"):
        print("\nloading checkpoint")
        model = RecurrentPPO.load(file_name, env=env, tensorboard_log=tensorboard_path)
        model.n_steps = ep_length
        model.n_envs = 1
        model.rollout_buffer.buffer_size = ep_length
        model.rollout_buffer.n_envs = 1
        model.rollout_buffer.reset()
    else:
        model = RecurrentPPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            n_steps=ep_length,
            batch_size=ep_length,
            n_epochs=1,
            tensorboard_log=tensorboard_path,
            gamma=0.99,
        )

    while True:
        model.learn(
            total_timesteps=ep_length * 100,
            callback=checkpoint_callback,
            tb_log_name=f"{goal}",
            reset_num_timesteps=True,
            # progress_bar=True,
        )

        model.save(drive + f"model/{goal}")
        deterministic_rewards, deterministic_lengths = evaluate_policy(
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

        with open(drive + "results.txt", "a") as f:
            f.write(
                f"Deterministic Result reward = {deterministic_rewards}, steps = {deterministic_lengths}"
            )
            f.write(
                f"Noneterministic Result reward = {episode_rewards}, steps = {episode_lengths}"
            )


train()
