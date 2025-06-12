import argparse
import os
import subprocess

import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    "-e",
    "--env",
    help="PATH to save result files",
    type=str,
    choices=["local", "server", "docker"],
    default="docker",
)
parser.add_argument("--cpus", type=int, default=os.cpu_count() - 1)
parser.add_argument("--date", type=str)

args = parser.parse_args()

if args.env == "local":
    dir_PATH = (
        "/Users/kishidayasuhiro/Documents/Master/RL4Quantum/Kishida_ver/Result/PID{}"
    )
    reward_PATH = dir_PATH + "/episode{}/reward.csv"

elif args.env == "server":
    dir_PATH = "/home/sogabelab/Documents/Kishida_ver/Result/PID{}"
    reward_PATH = dir_PATH + "/episode{}/reward.csv"

elif args.env == "docker":
    dir_PATH = "/home/RL4Quantum/Result/RL4Quantum/PID{}"
    reward_PATH = dir_PATH + "/episode{}/reward.csv"
    plt.switch_backend("TkAgg")


episode_nums = "find " + dir_PATH.format(0) + "/episode*" + " -type d | wc -l"
episode_nums = (
    subprocess.Popen(episode_nums, stdout=subprocess.PIPE, shell=True)
    .communicate()[0]
    .decode("utf-8")
)
episode_nums = int(episode_nums) - 1

for process in range(args.cpus):
    average_reward_list = []
    for episode in range(episode_nums):
        reward = pd.read_csv(reward_PATH.format(process, episode), header=None)
        reward = reward.mean()
        average_reward_list.append(reward)

    x = list(range(episode_nums))
    y = average_reward_list

    fig, ax = plt.subplots()

    ax.plot(x, y)
    ax.set_xlabel("Episode number")
    ax.set_ylabel("Average reward")
    ax.set_title("Reward curve")

    plt.savefig("Notes/{}/Result_reward_PID{}.png".format(args.date, process))
    plt.cla()
    plt.close()
