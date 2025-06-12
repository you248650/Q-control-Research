import csv
import os

import gym
import numpy as np
import torch
from gym import spaces
from qutip import *

from custom_gym_env.envs.double_harmonic_oscillator import *
from custom_gym_env.envs.smesolve import *

dir_PATH = "/home/RL4Quantum/Result/RL4Quantum/PID{}/episode{}"
observation_PATH = dir_PATH + "/observation.csv"
action_PATH = dir_PATH + "/action.csv"
reward_PATH = dir_PATH + "/reward.csv"
terminal_PATH = dir_PATH + "/terminal.csv"


class DoubleHarmonicOscillatorEnv(gym.Env):
    def __init__(self, process_id, dim):
        self.process_id = process_id
        self.dim = dim

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.min_action = -5
        self.max_action = 5
        self.min_measurement_current = -np.inf
        self.max_measurement_current = np.inf

        self.action_space = spaces.Box(
            low=self.min_action, high=self.max_action, shape=(1,), dtype=np.float64
        )
        self.observation_space = spaces.Box(
            low=self.min_measurement_current,
            high=self.max_measurement_current,
            shape=(1,),
            dtype=np.float64,
        )
        self.oscillator = DoubleHarmonicOscillator(self.dim)

        self.simulator = Simulator(self.process_id, self.dim)

        self.episode_count = -1

        self.PF_nums = self.simulator.PF_nums

    def reset(self, seed=None):
        self.episode_count += 1
        print(f"### PID: {self.process_id}  episode: {self.episode_count} ###")
        self.stepcount = 0

        self.rho = self.oscillator.Thermal_State()

        self.PF_rho = torch.stack(
            [self.oscillator.Thermal_State() for _ in range(self.PF_nums)]
        )

        os.makedirs(dir_PATH.format(self.process_id, self.episode_count), exist_ok=True)

        return np.zeros(1)

    def step(self, action):
        self.rho, dQ = self.simulator.SMEsolve(
            self.rho, action, self.dim, mode="SME simulator"
        )

        est_rho, self.PF_rho = self.simulator.Particle_Filter(self.PF_rho, action, dQ)

        est_dQ = self.simulator.SMEsolve(
            est_rho,
            action,
            self.PF_nums,
            mode="Measurement current for est_rho",
        )

        reward = -abs(est_dQ - 3**2)

        if self.stepcount == 1000:
            done = True
        else:
            done = False

        with open(
            observation_PATH.format(self.process_id, self.episode_count),
            "a",
        ) as f:
            save = csv.writer(f)
            save.writerow(np.reshape(np.array(self.rho.cpu()), (-1)))

        with open(
            action_PATH.format(self.process_id, self.episode_count),
            "a",
        ) as f:
            save = csv.writer(f)
            save.writerow(np.reshape(np.array(action), (-1)))

        with open(
            reward_PATH.format(self.process_id, self.episode_count),
            "a",
        ) as f:
            save = csv.writer(f)
            save.writerow(np.reshape(np.array(reward), (-1)))

        with open(
            terminal_PATH.format(self.process_id, self.episode_count),
            "a",
        ) as f:
            save = csv.writer(f)
            save.writerow(np.reshape(np.array(done), (-1)))

        self.stepcount += 1

        return np.array([est_dQ]), reward, done, {}

    def seed(self, seed=None):
        return np.random.seed(seed)

    def render(self, mode="console", close=False):
        print("done")

    def close(self):
        pass
