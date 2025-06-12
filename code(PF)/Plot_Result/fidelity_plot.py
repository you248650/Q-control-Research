import argparse
import csv
import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np
from qutip import *

parser = argparse.ArgumentParser()
parser.add_argument(
    "-e",
    "--env",
    type=str,
    help="PATH to save result files",
    choices=["local", "server", "docker"],
    default="docker",
)
parser.add_argument("--cpus", type=int, default=os.cpu_count() - 1)
parser.add_argument("--dim", type=int)
parser.add_argument("--date", type=str)

args = parser.parse_args()

if args.env == "local":
    dir_PATH = (
        "/Users/kishidayasuhiro/Documents/Master/RL4Quantum/Kishida_ver/Result/PID{}"
    )
    rho_PATH = dir_PATH + "/episode{}/observation.csv"

elif args.env == "server":
    dir_PATH = "/home/sogabelab/Documents/Kishida_ver/Result/PID{}"
    rho_PATH = dir_PATH + "/episode{}/observation.csv"

elif args.env == "docker":
    dir_PATH = "/home/RL4Quantum/Result/RL4Quantum/PID{}"
    rho_PATH = dir_PATH + "/episode{}/observation.csv"
    plt.switch_backend("TkAgg")


def cat_state():
    alpha = 2.0

    N = args.dim

    psi_plus = coherent(N, alpha)
    psi_minus = coherent(N, -alpha)

    cat_state = (psi_plus + psi_minus).unit()
    cat_state = cat_state * cat_state.dag()

    return cat_state


episode_nums = "find " + dir_PATH.format(0) + "/episode*" + " -type d | wc -l"
episode_nums = (
    subprocess.Popen(episode_nums, stdout=subprocess.PIPE, shell=True)
    .communicate()[0]
    .decode("utf-8")
)
episode_nums = int(episode_nums) - 1

for process in range(args.cpus):
    fidelity_list = []
    for episode in range(episode_nums):
        with open(rho_PATH.format(process, episode_nums), "r") as f:
            reader = csv.reader(f)
            rho = [row for row in reader]

        rho = np.reshape(rho[-1], (args.dim, args.dim))
        rho = Qobj(rho)
        if isket(rho):
            rho = ket2dm(rho)

        cat_state_ = cat_state()

        fidelity_ = fidelity(rho, cat_state) ** 100
        fidelity_list.append(fidelity_)

        x = list(range(episode_nums))
        y = fidelity_list

        fig, ax = plt.subplots()

        ax.plot(x, y)
        ax.set_xlabel("Episode number")
        ax.set_ylabel("Fidelity")
        ax.set_title("Fidelity curve")

        plt.savefig("Notes/{}/Result_fidelity_PID{}.png".format(args.date, process))
        plt.cla()
        plt.close()
