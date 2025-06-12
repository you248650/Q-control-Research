import argparse
import os

import torch
from custom_gym_env.envs.double_quantum_environment import DoubleHarmonicOscillatorEnv
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv

parser = argparse.ArgumentParser()
parser.add_argument("--cpus", type=int, default=os.cpu_count() - 1)
parser.add_argument("--dim", type=int)
parser.add_argument("--date", type=str)
args = parser.parse_args()


def make_env(env_id, process, seed=0):
    def _init():
        env = DoubleHarmonicOscillatorEnv(process_id=process, dim=args.dim)
        env.seed(process + seed)
        return env

    set_random_seed(seed)
    return _init


def main():
    env_id = "DoubleHarmonicOscillatorEnv-v0"
    env = SubprocVecEnv([make_env(env_id, i) for i in range(args.cpus)])

    # Hyperparameters
    pi = [512, 256, 128]
    vf = [512, 256, 128]
    verbose = 1
    learning_rate = 1e-5
    batch_size = 100
    n_steps = 1000
    episodes = 1000
    total_timesteps = n_steps * episodes * args.cpus

    policy_kwargs = dict(net_arch=[dict(pi=pi, vf=vf)])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=verbose,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_steps=n_steps,
        device=device,
    )

    # Make Logfile
    log_path = "Notes/{}".format(args.date)
    new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    # Train the model
    model.learn(total_timesteps=total_timesteps)

    new_logger.close()

    model.save("Result/RL4Quantum/PPO_even_thermal")

    env.close()


if __name__ == "__main__":
    main()
