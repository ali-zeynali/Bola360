from matplotlib import pyplot as plt

from Bola3d import *
from DDP import *
from DDPOnline import *
from HeadMoves import *
from Video import *


def calc_reward(solution, buffer, download_time, delta, probs, video, gamma):
    """

    :param solution:
    :param buffer:
    :param download_time:
    :param delta:
    :param probs:
    :param video:
    :param gamma:
    :return:
        r: expected reward of that download
        buffer: buffer level after downloading ends
        y: rebuffering time
    """
    ddp_n = 0
    for m in solution:
        if m > 0:
            ddp_n += 1
    y = max(download_time - buffer, 0)
    # buffer = max(buffer - download_time, 0) + ddp_n * video.delta
    buffer = buffer - download_time + y + ddp_n * video.delta
    expected_vals_dp = 0
    for i in range(D):
        expected_vals_dp += probs[i] * video.values[solution[i]]
    r0 = gamma * (delta) + expected_vals_dp
    if ddp_n > 0:
        return r0, buffer, y
    else:
        return 0, buffer, y


def get_total_size(set_of_actions, video):
    """

    :param set_of_actions: array size D, showing the bitrates of tiles which we are going to download
    :param video:
    :return:
        total_size: size of downloading segments in bits
        n: number of selected segments (non empty tiles) to download
    """
    total_size = 0
    n = 0
    for a in set_of_actions:
        total_size += video.sizes[a]
        if a > 0:
            n += 1
    return total_size, n


def convert_action_to_rates(actions, sizes, delta):
    rates = []
    for action in actions:
        rates.append(sizes[action] / delta)

    return rates


def plot_bandwidth(changes, bola, dt, final_time, bitrates, V, D):
    fig = plt.figure(0)
    x = []
    y = []
    t = 0
    indx = 0
    changes = changes.copy()
    while t < final_time:
        while changes[indx][0] < t:
            indx += 1
        x.append(t)
        y.append(changes[indx][1] / D)
        t += dt
    plt.plot(x, y, label="Network", linewidth=1.5)

    # Bola algorithm - avg
    x = []
    y = []
    t = 0
    last_played_time = 0
    indx = 0
    while t < final_time:

        while t < bola[indx][0]:
            x.append(t)
            y.append(0)
            t += dt
        x.append(t)
        y.append(np.average(bola[indx][1]))

        x.append(t + delta)
        y.append(np.average(bola[indx][1]))

        t += delta
        indx += 1

    plt.plot(x, y, label="Avg(Bola)", linewidth=1.5)

    # Bola algorithm - max
    x = []
    y = []
    t = 0
    last_played_time = 0
    indx = 0
    while t < final_time:

        while t < bola[indx][0]:
            x.append(t)
            y.append(0)
            t += dt
        x.append(t)
        y.append(np.max(bola[indx][1]))

        x.append(t + delta)
        y.append(np.max(bola[indx][1]))

        t += delta
        indx += 1

    plt.plot(x, y, label="Max(Bola)", linewidth=1.5)

    for br in bitrates:
        plt.plot([0, final_time], [br, br], label='_nolegend_', linestyle='dashed', color="gray", linewidth=0.7)
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Bitrate (Mbps)")
    plt.title("V(Bola): {0}".format(V))
    plt.savefig("figures/bandwidth_v_{0}.png".format(V), dpi=600)


ratio = 2
N = 20
D = 4

buffer_size = 8  # number of segments fit into buffer, unit: integer
delta = 5
gamma = 0
t_0 = delta / 10
wait_time = t_0

data_path = "dataset/bandwidth/4G/4G_BWData.json"
# data_path = "fix_bandwidth.json"

bandwidth_error = 0.10
sizes = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]) * delta / 2
sizes = np.array([0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12]) * delta

sizes = np.array([0, 1, 5, 7, 9, 11]) * delta
M = len(sizes) - 1
# v_coeff = buffer_size / (buffer_size - D)
v_coeff = 1
# values = np.array([i for i in range(M)]) * delta * 50/ 2
# for i in range(len(values)):
#     values[i] = np.power(values[i], 2)
values = np.array([0 if x == 0 else np.log(x / sizes[1]) for x in sizes])

video = Video(N, delta, D, values, sizes, buffer_size)
bandwidth = Bandwidth(data_path, error_rate=bandwidth_error)
nav_graph_path = "dataset/headmovement/navG1.json"
headMovements = HeadMoves(N, D, path=nav_graph_path)  # TODO

# bola3d = Bola3d(video, gamma, v_coeff)
ddp = DDP(video, buffer_size, bandwidth, gamma, t_0)
# ddp_online = DDPOnline(video,buffer_size, bandwidth, gamma, t_0)


print("Calculating optimal offline")
ddp.train(headMovements.get_all_probs())
optimal_offline = ddp.get_optimal_reward()
print("Offline: ", optimal_offline)

bola_performance = []
ddpO_performance = []
bola_solutions = []
ddp_solutions = []
max_bands = np.arange(50, 500, 0.2)
# times_bola = []
# times_ddp =
buffer_levels_bola = []
buffer_levels_ddp = []
time_levels_bola = []
time_levels_ddp = []

#
# for max_band in max_bands:
#     time_bola = 0
#     buffer_bola = 0
#     reward_bola = 0
#     bola_solutions[max_band] = []
#     ddp_solutions[max_band] = []
#     buffer_levels_bola[max_band] = []
#     buffer_levels_ddp[max_band] = []
#     time_levels_bola[max_band] = []
#     time_levels_ddp[max_band] = []


time_dp = 0
buffer_dp = 0
reward_dp = 0
bola3d = Bola3d(video, gamma, v_coeff)
ddp_online = DDPOnline(video, buffer_size, bandwidth, gamma, t_0)

total_reward_bola = 0
total_reward_ddp_online = 0

# DDP Online
# n = 0
# while True:
#     if n >= N:
#         break
#     time_levels_ddp.append(time_dp)
#     buffer_levels_ddp.append(buffer_dp)
#     print("ddP: n: ", n)
#     probs = headMovements.get_probs(n)
#     #        print("probs: ", probs)
#     #        print("time: ", time_dp)
#     # print("getting DP action")
#     ddp_action = ddp_online.get_action(probs, time_dp, buffer_dp, reward_dp)
#     #        print("n: " ,n ,"\tDDP-On action: ", ddp_action)
#     total_size = get_total_size(ddp_action, video)
#     if np.sum(ddp_action) > 0:
#         download_time_dp = bandwidth.download_time(total_size, time_dp)
#         n += 1
#     else:
#         download_time_dp = wait_time
#     time_dp += download_time_dp
#     # r_dp, buffer_dp, y  = calc_reward(ddp_action, buffer_dp, download_time_dp, delta, probs, video, gamma)  # TODO
#     # r_dp, buffer_dp, y = calc_reward(ddp_action, buffer_dp, download_time_dp, delta, probs, video, gamma)  # TODO
#     ddp_solutions.append(ddp_action)
#     # reward_dp = + r_dp
# # times_ddp.append(time_dp)

bola_play_rates = []
time_bola = 0
buffer_bola = 0  # unit: time
reward_bola = 0
rebuffer_bola = 0

# bola algorithm
n = 0

while True:
    if n >= N:
        break
    time_levels_bola.append(time_bola)
    buffer_levels_bola.append(buffer_bola)
    # print("Bola n: ", n)
    probs = headMovements.head_movement(n)

    # print("getting bola action")
    bola_action = bola3d.get_action(probs)
    bola_solutions.append(bola_action)
    # print("getting DP action")

    if np.sum(bola_action) > 0:
        total_size, download_n = get_total_size(bola_action, video)
        download_time_bola = bandwidth.download_time(total_size, time_bola)
        bola3d.take_action(bola_action, n, time_bola)
        bola_play_rates.append((time_bola + download_time_bola, convert_action_to_rates(bola_action, sizes, delta)))
        # print("Downloaded : {0} and took {1} seconds".format(bola_action, download_time_bola))
        n += 1
    else:
        download_n = 0
        download_time_bola = wait_time
        bola3d.take_action(bola_action, n, time_bola)
    time_bola += max(download_time_bola, (download_n - buffer_size) * delta + buffer_bola - download_time_bola)

    reward_bola, buffer_bola, reb = calc_reward(bola_action, buffer_bola, download_time_bola, delta, probs, video,
                                           gamma)
    rebuffer_bola += reb
    total_reward_bola += reward_bola
print("Bola performance: {0}".format(total_reward_bola / (time_bola + t_0)))
# bola_performance.append(reward_bola / time_bola)
# ddpO_performance.append(reward_dp / time_dp)

# print("Bola: r = {0}, time={1}, b={2}, r_0={3}".format(reward_bola/ time_bola,time_bola, buffer_bola, reward_bola))
# print("DPP: r = {0}, time={1}, b={2}, r_0={3}".format(reward_dp/ time_dp,time_dp, buffer_dp, reward_dp))


# plt.plot(max_bands, bola_performance, label="Bola360")
# plt.plot(max_bands, ddpO_performance, label="DDP-Online")
# plt.legend()
# plt.xlabel("Bandwidth's average")
# plt.title("Objective values of Bola360 vs DDP-Online")
# plt.savefig("bandwidth_change.png", dpi=600)

plot_bandwidth(bandwidth.throuput_changes, bola_play_rates, 0.1, max(time_bola, delta * N), sizes / delta, v_coeff, D)
