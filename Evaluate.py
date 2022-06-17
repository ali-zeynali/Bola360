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
    naive_half = read_dataset(path + "/naive_3.json")
    naive_full = read_dataset(path + "/naive_6.json")

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


def compare_rewards_val(number_of_samples=4, __print__=True):
    bola = []
    ddp = []
    naive1 = []
    naive_half = []
    naive_full = []

    for sample in range(number_of_samples):
        path = "results/sample_{0}".format(sample)
        rewards = get_attr(path, 'reward_val')
        bola.append(rewards['bola'])
        ddp.append(rewards['ddp'])
        naive1.append(rewards['naive1'])
        naive_half.append(rewards['naive_h'])
        naive_full.append(rewards['naive_f'])

    if __print__:
        print("\n<------\tComparing Value Rewards\t------>")
        print("Bola3d Average value: {0}".format(np.average(bola)))
        print("DP-Online Average value: {0}".format(np.average(ddp)))
        print("Naive1 Average value: {0}".format(np.average(naive1)))
        print("Naive-half Average value: {0}".format(np.average(naive_half)))
        print("Naive-full Average value: {0}".format(np.average(naive_full)))

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
    downloaded_bitrates = []
    for i in range(len(solution) - 1):
        n = 0
        for a in solution[i]:
            if a > 0:
                n += 1
        if n < 1:
            continue
        tiles.append(n)
        available_time.append(time[i + 1])
        downloaded_bitrates.append(solution[i])
    return available_time, tiles, downloaded_bitrates


def get_playing_bitrates(solution, buffer, time, actual_headmovement, bitrates, delta, values, dt=0.05,
                         __start_time__=0):
    available_time, tiles, downloaded_bitrates = get_available_times(solution, time)

    watching_bitrate = []
    time_slots = []
    getting_values = []

    playing_segment = -1
    portion_remained = 0
    t = __start_time__
    buffer_level = 0
    play_rate = 0
    downloaded_segment = -1

    rebuff = 0
    while t < available_time[-1] or buffer_level > 0:

        # add downloaded segment to buffer
        if downloaded_segment < len(available_time) - 1 and t >= available_time[downloaded_segment + 1]:
            buffer_level = np.maximum(buffer[downloaded_segment + 1], 0)
            downloaded_segment += 1

        # go to next segment if available
        if portion_remained <= 0 and playing_segment < len(available_time) - 1 and t >= available_time[
            playing_segment + 1]:
            playing_segment += 1
            portion_remained = delta
            play_rate = tiles[playing_segment]

        watching_tile = actual_headmovement[playing_segment]

        # if segment has not downloaded yet
        if t < available_time[playing_segment]:
            rebuff += dt
            watching_bitrate.append(0)
            time_slots.append(t)
            getting_values.append(0)

        # if the watching tile is not in list of downloaded segments
        elif portion_remained > 0 and t >= available_time[playing_segment] and downloaded_bitrates[playing_segment][
            watching_tile] == 0:
            rebuff += dt
            watching_bitrate.append(
                -5)  # Just to make a difference between waiting to download the segment and downloading wrong tiles
            time_slots.append(t)
            getting_values.append(0)

        elif portion_remained > 0 and t >= available_time[playing_segment]:
            r_indx = downloaded_bitrates[playing_segment][watching_tile]
            r = bitrates[r_indx]
            watching_bitrate.append(r)
            time_slots.append(t)
            getting_values.append(values[r_indx])

        # if segment has not downloaded yet
        elif portion_remained <= 0 and t >= available_time[playing_segment]:
            rebuff += dt
            watching_bitrate.append(0)
            time_slots.append(t)
            getting_values.append(0)

        buffer_level -= play_rate * dt
        buffer_level = np.maximum(buffer_level, 0)
        portion_remained -= dt

        t += dt

    avg_wbr = get_average_watching_bitrate(watching_bitrate)
    avg_wv = np.average(getting_values)
    return rebuff, watching_bitrate, time_slots, avg_wbr, avg_wv


def real_evaluation(path_to_save, sample, delta, values, DDP=False, N1=True, Nh=True, Nf=True, __start_time__=0,
                    __print__=True):
    fig = plt.figure(int(np.random.random() * 10000))

    path = "results/sample_{0}".format(sample)

    meta_data = read_dataset(path + "/meta.json")
    actual_headmovement = meta_data['view']
    bitrates = np.array(meta_data['sizes']) / delta

    buffers = get_attr(path, 'buffer')
    times = get_attr(path, 'time')
    solution = get_attr(path, 'solution')

    bola_buff = buffers['bola']
    ddp_buff = buffers['ddp']
    naive1_buff = buffers['naive1']
    naive_half_buff = buffers['naive_h']
    naive_full_buff = buffers['naive_f']

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

    bola_rebuff, bola_y, bola_x, bola_avg_btr, bola_avg_wv = get_playing_bitrates(bola_solution, bola_buff, bola_time,
                                                                                  actual_headmovement, bitrates, delta,
                                                                                  values,
                                                                                  __start_time__=__start_time__)
    ddp_rebuff, ddp_y, ddp_x, ddp_avg_btr, ddp_avg_wv = get_playing_bitrates(ddp_solution, ddp_buff, ddp_time,
                                                                             actual_headmovement,
                                                                             bitrates, delta, values,
                                                                             __start_time__=__start_time__)
    naive1_rebuff, naive1_y, naive1_x, naive1_avg_btr, naive1_avg_wv = get_playing_bitrates(naive1_solution,
                                                                                            naive1_buff, naive1_time,
                                                                                            actual_headmovement,
                                                                                            bitrates, delta, values,
                                                                                            __start_time__=__start_time__)
    naivh_rebuff, naivh_y, naivh_x, naivh_avg_btr, naivh_avg_wv = get_playing_bitrates(naivh_solution, naive_half_buff,
                                                                                       naive_half_time,
                                                                                       actual_headmovement, bitrates,
                                                                                       delta, values,
                                                                                       __start_time__=__start_time__)
    naivf_rebuff, naivf_y, naivf_x, naivf_avg_btr, naivf_avg_wv = get_playing_bitrates(naivf_solution, naive_full_buff,
                                                                                       naive_full_time,
                                                                                       actual_headmovement, bitrates,
                                                                                       delta, values,
                                                                                       __start_time__=__start_time__)

    plt.plot(bola_x, bola_y, label="Bola360", linewidth=1.5)
    if DDP:
        plt.plot(ddp_x, ddp_y, label="DP-Online", linewidth=1.5)
    if N1:
        plt.plot(naive1_x, naive1_y, label="Naive-1", linewidth=1.5)
    if Nh:
        plt.plot(naivh_x, naivh_y, label="Naive-half", linewidth=1.5)
    if Nf:
        plt.plot(naivf_x, naivf_y, label="Naive-full", linewidth=1.5)

    for br in bitrates:
        plt.plot([np.min([bola_x[0], naivf_x[0], naive1_x[0], naivf_x[0]]),
                  np.max([bola_x[-1], naivf_x[-1], naive1_x[-1], naivf_x[-1]])], [br, br], label='_nolegend_',
                 linestyle='dashed', color="gray",
                 linewidth=0.7)

    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Playing bitrate (Mbps)")
    plt.savefig(path_to_save, dpi=600, bbox_inches='tight')

    if __print__:
        print("\n<------\tComparing Actual bitrates\t------>")
        print("Bola3d Average watching bitrate: {0:.1f}, rebuffering: {1:.1f}, avg v: {2:.1f}".format(bola_avg_btr,
                                                                                                      bola_rebuff,
                                                                                                      bola_avg_wv))
        print("DP-Online Average watching bitrate: {0:.1f}, rebuffering: {1:.1f}, avg v: {2:.1f}".format(ddp_avg_btr,
                                                                                                         ddp_rebuff,
                                                                                                         ddp_avg_wv))
        print("Naive1 Average watching bitrate: {0:.1f}, rebuffering: {1:.1f}, avg v: {2:.1f}".format(naive1_avg_btr,
                                                                                                      naive1_rebuff,
                                                                                                      naive1_avg_wv))
        print("Naive-half Average watching bitrate: {0:.1f}, rebuffering: {1:.1f}, avg v: {2:.1f}".format(naivh_avg_btr,
                                                                                                          naivh_rebuff,
                                                                                                          naivh_avg_wv))
        print("Naive-full Average watching bitrate: {0:.1f}, rebuffering: {1:.1f}, avg v: {2:.1f}".format(naivf_avg_btr,
                                                                                                          naivf_rebuff,
                                                                                                          naivf_avg_wv))
        print("\n<------\t ----------------- \t------>")


def get_average_watching_bitrate(watching_bitrate):
    wbr = [0 if x <= 0 else x for x in watching_bitrate]
    return np.average(wbr)


def make_buffer_lines(buffer, solution, time, delta, dt=0.05, __start_time__=0):
    available_time, tiles, _ = get_available_times(solution, time)

    x = []
    y = []

    playing_segment = -1
    portion_remained = 0
    t = __start_time__
    buffer_level = 0
    play_rate = 0
    downloaded_segment = -1
    while t < available_time[-1] or buffer_level > 0:
        x.append(t)
        y.append(buffer_level)
        t += dt
        # play next segment if available
        if portion_remained <= 0 and playing_segment < len(available_time) - 1 and t >= available_time[
            playing_segment + 1]:
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


def compare_buffers(path_to_save, sample, delta, DDP=False, __start_time__=0):
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

    bola_y, bola_x = make_buffer_lines(bola, bola_solution, bola_time, delta, __start_time__=__start_time__)
    ddp_y, ddp_x = make_buffer_lines(ddp, ddp_solution, ddp_time, delta, __start_time__=__start_time__)
    naive1_y, naive1_x = make_buffer_lines(naive1, naive1_solution, naive1_time, delta, __start_time__=__start_time__)
    naivh_y, naivh_x = make_buffer_lines(naive_half, naivh_solution, naive_half_time, delta,
                                         __start_time__=__start_time__)
    naivf_y, naivf_x = make_buffer_lines(naive_full, naivf_solution, naive_full_time, delta,
                                         __start_time__=__start_time__)

    plt.plot(bola_x, bola_y, label="Bola360", linewidth=1.5)
    if DDP:
        plt.plot(ddp_x, ddp_y, label="DP-Online", linewidth=1.5)
    plt.plot(naive1_x, naive1_y, label="Naive-1", linewidth=1.5)
    plt.plot(naivh_x, naivh_y, label="Naive-half", linewidth=1.5)
    plt.plot(naivf_x, naivf_y, label="Naive-full", linewidth=1.5)

    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Buffer Levels (s)")
    plt.savefig(path_to_save, dpi=600, bbox_inches='tight')

    return bola, ddp, naive1, naive_half, naive_full

# compare_rewards()
