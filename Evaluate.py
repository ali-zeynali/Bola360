import json

import numpy as np
from matplotlib import pyplot as plt


def read_dataset(path):
    with open(path) as reader:
        data = json.load(reader)
    return data


def get_attr(path, attr):
    bola = read_dataset(path + "/bola.json")
    ddp = read_dataset(path + "/DDP.json")
    naive1 = read_dataset(path + "/naive_1.json")
    naive_half = read_dataset(path + "/naive_2.json")
    naive_full = read_dataset(path + "/naive_4.json")

    rewards = {}
    rewards['bola'] = bola[attr]
    rewards['ddp'] = ddp[attr]
    rewards['naive1'] = naive1[attr]
    rewards['naive_h'] = naive_half[attr]
    rewards['naive_f'] = naive_full[attr]

    return rewards


def compare_rewards(number_of_samples=4, __print__=True):
    bola = []
    ddp = []
    naive1 = []
    naive_half = []
    naive_full = []

    for sample in range(number_of_samples):
        path = "results/sample_{0}".format(sample)
        rewards = get_attr(path, 'reward')
        bola.append(rewards['bola'])
        ddp.append(rewards['ddp'])
        naive1.append(rewards['naive1'])
        naive_half.append(rewards['naive_h'])
        naive_full.append(rewards['naive_f'])

    if __print__:
        print("\n<------\tComparing Rewards\t------>")
        print("Bola3d Average utility: {0}".format(np.average(bola)))
        print("DP-Online Average utility: {0}".format(np.average(ddp)))
        print("Naive1 Average utility: {0}".format(np.average(naive1)))
        print("Naive-half Average utility: {0}".format(np.average(naive_half)))
        print("Naive-full Average utility: {0}".format(np.average(naive_full)))

    return bola, ddp, naive1, naive_half, naive_full

    # fig = plt.figure(figsize=(10, 7))

    # Creating axes instance
    # ax = fig.add_axes([0, 0, 1, 1])

    # plt.boxplot(data)
    # ax.set_yticklabels(['Bola', 'DP-Online', 'Naive-1', 'Naive-half', 'Naive-full'])
    # plt.show()


def compare_rebuff(number_of_samples=4, __print__=True):
    bola = []
    ddp = []
    naive1 = []
    naive_half = []
    naive_full = []

    for sample in range(number_of_samples):
        path = "results/sample_{0}".format(sample)
        attr = get_attr(path, 'rebuff')
        bola.append(attr['bola'])
        ddp.append(attr['ddp'])
        naive1.append(attr['naive1'])
        naive_half.append(attr['naive_h'])
        naive_full.append(attr['naive_f'])

    if __print__:
        print("\n<------\tComparing Re-buffering\t------>")
        print("Bola3d Average rebuff: {0}".format(np.average(bola)))
        print("DP-Online Average rebuff: {0}".format(np.average(ddp)))
        print("Naive1 Average rebuff: {0}".format(np.average(naive1)))
        print("Naive-half Average rebuff: {0}".format(np.average(naive_half)))
        print("Naive-full Average rebuff: {0}".format(np.average(naive_full)))

    return bola, ddp, naive1, naive_half, naive_full


def get_available_times(solution, time):
    available_time = []
    tiles = []
    for i in range(len(solution) - 1):
        n = 0
        for a in solution[i]:
            if a > 0:
                n += 1
        if n < 1:
            continue
        tiles.append(n)
        available_time.append(time[i + 1])
    return available_time, tiles


def make_buffer_lines(buffer, solution, time, delta, dt=0.05):
    available_time, tiles = get_available_times(solution, time)

    x = []
    y = []

    playing_segment = -1
    portion_remained = 0
    t = 0
    buffer_level = 0
    play_rate = 0
    downloaded_segment = -1
    while t < available_time[-1] or buffer_level > 0:
        x.append(t)
        y.append(buffer_level)
        t += dt
        # play next segment if available
        if portion_remained <= 0 and playing_segment < len(available_time) - 1 and t >= available_time[playing_segment + 1]:
            playing_segment += 1
            portion_remained = delta
            play_rate = tiles[playing_segment]
        buffer_level -= play_rate * dt
        buffer_level = np.maximum(buffer_level, 0)
        portion_remained -= dt

        # add downloaded segment to buffer
        if downloaded_segment < len(available_time) - 1 and t >= available_time[downloaded_segment + 1]:
            buffer_level = np.maximum(buffer[downloaded_segment + 1], 0)
            downloaded_segment += 1
    return y, x


def compare_buffers(path_to_save, sample, delta, DDP=False):

    fig = plt.figure(int(np.random.random() * 10000))
    path = "results/sample_{0}".format(sample)
    buffers = get_attr(path, 'buffer')
    times = get_attr(path, 'time')
    solution = get_attr(path, 'solution')

    bola = buffers['bola']
    ddp = buffers['ddp']
    naive1 = buffers['naive1']
    naive_half = buffers['naive_h']
    naive_full = buffers['naive_f']

    bola_time = times['bola']
    ddp_time = times['ddp']
    naive1_time = times['naive1']
    naive_half_time = times['naive_h']
    naive_full_time = times['naive_f']

    bola_solution = solution['bola']
    ddp_solution = solution['ddp']
    naive1_solution = solution['naive1']
    naivh_solution = solution['naive_h']
    naivf_solution = solution['naive_f']

    bola_y, bola_x = make_buffer_lines(bola, bola_solution, bola_time, delta)
    ddp_y, ddp_x = make_buffer_lines(ddp, ddp_solution, ddp_time, delta)
    naive1_y, naive1_x = make_buffer_lines(naive1, naive1_solution, naive1_time, delta)
    naivh_y, naivh_x = make_buffer_lines(naive_half, naivh_solution, naive_half_time, delta)
    naivf_y, naivf_x = make_buffer_lines(naive_full, naivf_solution, naive_full_time, delta)

    plt.plot(bola_x, bola_y, label="Bola360", linewidth=1.5)
    if DDP:
        plt.plot(ddp_x, ddp_y, label="DP-Online", linewidth=1.5)
    plt.plot(naive1_x, naive1_y, label="Naive-1", linewidth=1.5)
    plt.plot(naivh_x, naivh_y, label="Naive-half", linewidth=1.5)
    plt.plot(naivf_x, naivf_y, label="Naive-full", linewidth=1.5)

    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Buffer Levels (s)")
    plt.savefig(path_to_save, dpi=600)

    return bola, ddp, naive1, naive_half, naive_full

# compare_rewards()
