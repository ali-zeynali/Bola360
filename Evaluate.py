import json
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

def read_dataset(path):
    with open(path) as reader:
        data = json.load(reader)
    return data


def get_rewards(path):
    bola = read_dataset(path + "/bola.json")
    ddp = read_dataset(path + "/DDP.json")
    naive1 = read_dataset(path + "/naive_1.json")
    naive_half = read_dataset(path + "/naive_2.json")
    naive_full = read_dataset(path + "/naive_4.json")

    rewards = {}
    rewards['bola'] = bola['reward']
    rewards['ddp'] = ddp['reward']
    rewards['naive1'] = naive1['reward']
    rewards['naive_h'] = naive_half['reward']
    rewards['naive_f'] = naive_full['reward']

    return rewards


def compare_rewards(number_of_samples=4):
    bola = []
    ddp = []
    naive1 = []
    naive_half = []
    naive_full = []

    for sample in range(number_of_samples):
        path = "results/sample_{0}".format(sample)
        rewards = get_rewards(path)
        bola.append(rewards['bola'])
        ddp.append(rewards['ddp'])
        naive1.append(rewards['naive1'])
        naive_half.append(rewards['naive_h'])
        naive_full.append(rewards['naive_f'])

    data = [bola, ddp, naive1, naive_half, naive_full]


    print("Bola3d Average utility: {0}".format(np.average(bola)))
    print("DP-Online Average utility: {0}".format(np.average(ddp)))
    print("Naive1 Average utility: {0}".format(np.average(naive1)))
    print("Naive-half Average utility: {0}".format(np.average(naive_half)))
    print("Naive-full Average utility: {0}".format(np.average(naive_full)))

    # fig = plt.figure(figsize=(10, 7))

    # Creating axes instance
    # ax = fig.add_axes([0, 0, 1, 1])

    
    # plt.boxplot(data)
    # ax.set_yticklabels(['Bola', 'DP-Online', 'Naive-1', 'Naive-half', 'Naive-full'])
    # plt.show()


compare_rewards()

