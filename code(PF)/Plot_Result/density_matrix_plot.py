import argparse
import csv
import os
import subprocess

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
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


def plot_fock_distribution(rho):
    axes[0].bar(
        np.arange(0, 10), np.real(rho.diag())[:10], color="green", alpha=0.6, width=0.8
    )

    axes[0].set_xlim(-0.5, 10 + 0.5)
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("Fock number", fontsize=12)
    axes[0].set_ylabel("Occupation probability", fontsize=12)
    axes[0].set_title("Fock distribution", fontsize=12)


def plot_wigner(rho):
    W0 = wigner(rho, xvec, xvec, method="iterative")
    W, yvec = W0 if type(W0) is tuple else (W0, xvec)
    nrm = mpl.colors.Normalize(-W0.max(), W0.max())

    axes[1].contourf(xvec, xvec, W, 100, cmap=cm.RdBu, norm=nrm)

    if xvec is not yvec:
        axes[1].set_ylim(xvec.min(), xvec.max())

    axes[1].set_xlabel(r"$\rm{Re}(\alpha)$", fontsize=12)
    axes[1].set_ylabel(r"$\rm{Im}(\alpha)$", fontsize=12)
    axes[1].set_title("Wigner function", fontsize=12)


episode_nums = "find " + dir_PATH.format(0) + "/episode*" + " -type d | wc -l"
episode_nums = (
    subprocess.Popen(episode_nums, stdout=subprocess.PIPE, shell=True)
    .communicate()[0]
    .decode("utf-8")
)
episode_nums = int(episode_nums) - 1

for process in range(args.cpus):
    with open(rho_PATH.format(process, episode_nums), "r") as f:
        reader = csv.reader(f)
        rho = [row for row in reader]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    xvec = np.linspace(-5, 5, 500)

    rho = np.reshape(rho[-1], (args.dim, args.dim))
    rho = Qobj(rho)
    if isket(rho):
        rho = ket2dm(rho)

    # Fock distribution
    plot_fock_distribution(rho)

    # Wigner function
    plot_wigner(rho)

    plt.savefig("Notes/{}/Result_rho_PID{}.png".format(args.date, process))
    plt.cla()
    plt.close()
