import glob
import os
import shutil

from matplotlib import pyplot as plt

from Bola3d import *
from DDP import *
from DDPOnline import *
from HeadMoves import *
from Naive import *
from Video import *


def calc_reward(solution, segment, delta, probs, video, gamma, buffer_array, available_time, time,
                rebuffer):
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

    played_time = time - rebuffer
    available_buffer = max(segment * delta, 0)
    played_time = min(played_time, available_buffer)
    last_played_segment = int(played_time / delta)
    for i in range(last_played_segment):
        buffer_array[i] = 0

    last_available = 0 - delta
    y = None
    for i in range(segment):
        if available_time[i] > 0:
            last_available = available_time[i]

        if buffer_array[i] > 0:
            y = 0
            break

    if y is None:
        y = time - last_available - delta
        x = 1

    # y = max(download_time - buffer, 0) #TODO, this line has a bug
    # buffer = max(buffer - download_time, 0) + ddp_n * video.delta
    # buffer = buffer - download_time + y + ddp_n * video.delta  # TODO, this line has a bug
    buffer = np.sum(buffer_array) * delta
    expected_vals_dp = 0
    for i in range(D):
        expected_vals_dp += probs[i] * video.values[solution[i]]
    r0 = gamma * (delta) + expected_vals_dp
    if ddp_n > 0:
        return r0, buffer, y, buffer_array
    else:
        return 0, buffer, y, buffer_array


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


def save_meta(N, D, buffer_size, delta, gamma, t_0, wait_time, sizes, b_error, v_coeff, actual_view,
              path='results/meta.json'):
    data = {}
    data['N'] = N
    data['D'] = D
    data['b_max'] = buffer_size
    data['delta'] = delta
    data['gamma'] = gamma
    data['t_0'] = t_0
    data['wait'] = wait_time
    data['sizes'] = [float(x) for x in sizes]
    data['b_error'] = b_error
    data['V'] = v_coeff
    data['view'] = [int(x) for x in actual_view]
    with open(path, 'w') as writer:
        json.dump(data, writer)


def get_sample_paths():
    folders = glob.glob("results/sample*")
    paths = []
    for folder in folders:
        index = folder.split("_")[1]
        paths.append(["results/sample_{0}".format(index), int(index)])
    return paths


def read_meta(path):
    with open(path) as reader:
        data = json.load(reader)
    return data


# ratio = 2
N = 30
D = 4

buffer_size = 8  # number of segments fit into buffer, unit: integer
delta = 5
gamma = 2
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

headMovements = HeadMoves(N, D, path=nav_graph_path)

__run_optimal_offline__ = False
#############################
#############################
###   OPTIMAL OFFLINE     ###
#############################
#############################
if __run_optimal_offline__:
    samples = 1
    for i in range(samples):
        actual_movement = headMovements.get_sample_actual_view()
        headMovements.set_actual_views(actual_movement)
        sample_path = "results/sample_{0}/".format(i)
        try:
            shutil.rmtree(sample_path)
        except Exception:
            pass

        os.mkdir(sample_path)

        save_meta(N, D, buffer_size, delta, gamma, t_0, wait_time, sizes, bandwidth_error, v_coeff, actual_movement,
                  path=sample_path + "meta.json")

        # bola3d = Bola3d(video, gamma, v_coeff)
        ddp = DDP(video, buffer_size, bandwidth, gamma, t_0)
        # ddp_online = DDPOnline(video,buffer_size, bandwidth, gamma, t_0)

        print("Sample: {0}\tCalculating optimal offline".format(i))
        ddp.train(headMovements.get_all_probs())
        optimal_offline = ddp.get_optimal_reward()
        ddp.save_info(path=sample_path + "offline.json")
        # print("Offline: ", optimal_offline)

__run_bola__ = False
#############################
#############################
###         BOLA          ###
#############################
#############################

if __run_bola__:
    sample_paths = get_sample_paths()
    for [path, sample] in sample_paths:
        print("Bola starts work for sample {0}".format(sample))
        meta = read_meta(path + "/meta.json")
        N = meta['N']
        D = meta['D']
        buffer_size = meta['b_max']
        delta = meta['delta']
        gamma = meta['gamma']
        t_0 = meta['t_0']
        wait_time = meta['wait']
        sizes = np.array(meta['sizes'])
        bandwidth_error = meta['b_error']
        v_coeff = meta['V']
        actual_movement = meta['view']

        video = Video(N, delta, D, values, sizes, buffer_size)
        bandwidth = Bandwidth(data_path, error_rate=bandwidth_error)
        nav_graph_path = "dataset/headmovement/navG1.json"

        headMovements = HeadMoves(N, D, path=nav_graph_path)
        headMovements.set_actual_views(actual_movement)

        # bola_performance = []
        bola_solutions = []
        buffer_levels_bola = []
        available_content = [-1 for _ in range(N)]
        buffer_array_bola = [0 for _ in range(N)]
        time_levels_bola = []

        bola_play_rates = []
        time_bola = 0
        buffer_bola = 0  # unit: time
        reward_bola = 0
        rebuffer_bola = 0
        total_reward_bola = 0

        bola3d = Bola3d(video, gamma, v_coeff, buffer_size)

        # bola algorithm
        n = 0

        while True:
            if n >= N:
                break
            time_levels_bola.append(time_bola)
            buffer_levels_bola.append(buffer_bola)
            # print("Bola n: ", n)
            probs = headMovements.get_pose_probs(n)

            # print("getting bola action")
            bola_action = bola3d.get_action(probs)
            # print("Bola action: {0}".format(bola_action))
            bola_solutions.append(bola_action)
            # print("getting DP action")

            if np.sum(bola_action) > 0:
                total_size, download_n = get_total_size(bola_action, video)
                download_time_bola = bandwidth.download_time(total_size, time_bola)
                buffer_array_bola[n] = download_n
                bola3d.take_action(bola_action, n, time_bola)
                bola_play_rates.append(
                    (time_bola + download_time_bola, convert_action_to_rates(bola_action, sizes, delta)))
                # print("Downloaded : {0} and took {1} seconds".format(bola_action, download_time_bola))
            else:
                download_n = 0
                download_time_bola = wait_time
                bola3d.take_action(bola_action, n, time_bola)

            time_bola += max(download_time_bola, (download_n - buffer_size) * delta + buffer_bola - download_time_bola)

            if np.sum(bola_action) > 0:
                available_content[n - 1] = time_bola

            reward_bola, buffer_bola, reb, buffer_array_bola = calc_reward(bola_action, n, delta, probs, video,
                                                                           gamma,
                                                                           buffer_array_bola, available_content,
                                                                           time_bola,
                                                                           rebuffer_bola)
            if np.sum(bola_action) > 0:
                n += 1

            bola3d.set_buffer(buffer_bola / delta)

            rebuffer_bola += reb
            total_reward_bola += reward_bola
            # print("Buffer: {0}".format(buffer_bola / delta))
            # print(("Buffer-bola: {0}".format(bola3d.buffer)))
        # print("Bola performance: {0}".format(total_reward_bola / (time_bola + t_0)))

        result = {}
        result['solution'] = bola_solutions
        result['time'] = time_levels_bola
        result['buffer'] = [int(x) for x in buffer_levels_bola]
        result['reward'] = float(total_reward_bola / (time_bola + t_0))
        result['final_time'] = float(time_bola + t_0)
        result['playing_br'] = bola_play_rates
        result['rebuff'] = float(rebuffer_bola)

        with open(path + "/bola.json", 'w') as writer:
            json.dump(result, writer)

        print("Bola finished work for sample {0}".format(sample))

__run_ddp__ = False
#############################
#############################
###         DDP          ###
#############################
#############################

if __run_ddp__:
    sample_paths = get_sample_paths()
    for [path, sample] in sample_paths:
        print("DDP starts work for sample {0}".format(sample))
        meta = read_meta(path + "/meta.json")
        N = meta['N']
        D = meta['D']
        buffer_size = meta['b_max']
        delta = meta['delta']
        gamma = meta['gamma']
        t_0 = meta['t_0']
        wait_time = meta['wait']
        sizes = np.array(meta['sizes'])
        bandwidth_error = meta['b_error']
        v_coeff = meta['V']
        actual_movement = meta['view']

        video = Video(N, delta, D, values, sizes, buffer_size)
        bandwidth = Bandwidth(data_path, error_rate=bandwidth_error)
        nav_graph_path = "dataset/headmovement/navG1.json"

        headMovements = HeadMoves(N, D, path=nav_graph_path)
        headMovements.set_actual_views(actual_movement)

        time_dp = 0
        buffer_dp = 0
        reward_dp = 0
        rebuffer_dp = 0
        ddp_solutions = []
        buffer_levels_ddp = []
        available_content = [-1 for _ in range(N)]
        buffer_array_ddp = [0 for _ in range(N)]
        time_levels_ddp = []
        ddpO_performance = []

        ddp_play_rates = []
        ddp_online = DDPOnline(video, buffer_size, bandwidth, gamma, t_0)
        total_reward_ddp_online = 0
        # DDP Online
        n = 0
        while True:
            if n >= N:
                break
            time_levels_ddp.append(time_dp)
            buffer_levels_ddp.append(buffer_dp)

            probs = headMovements.get_pose_probs(n)

            ddp_action = ddp_online.get_action(probs, time_dp, buffer_dp, reward_dp)
            ddp_solutions.append(ddp_action)

            total_size, download_n = get_total_size(ddp_action, video)
            if np.sum(ddp_action) > 0:
                download_time_dp = bandwidth.download_time(total_size, time_dp)
                buffer_array_ddp[n] = download_n
                ddp_online.take_action(ddp_action, n, time_dp)
                ddp_play_rates.append((time_dp + download_time_dp, convert_action_to_rates(ddp_action, sizes, delta)))
            else:
                download_n = 0
                download_time_dp = wait_time
                ddp_online.take_action(ddp_action, n, time_dp)

            time_dp += max(download_time_dp, (download_n - buffer_size) * delta + buffer_dp - download_time_dp)

            if np.sum(ddp_action) > 0:
                available_content[n - 1] = time_dp

            reward_ddp, buffer_dp, reb, buffer_array_ddp = calc_reward(ddp_action, n, delta, probs, video,
                                                                       gamma, buffer_array_ddp, available_content,
                                                                       time_dp, rebuffer_dp)

            if np.sum(ddp_action) > 0:
                n += 1

            ddp_online.set_buffer(buffer_dp / delta)
            rebuffer_dp += reb
            reward_dp += reward_ddp

        result = {}
        result['solution'] = ddp_solutions
        result['time'] = time_levels_ddp
        result['buffer'] = [int(x) for x in buffer_levels_ddp]
        result['reward'] = float(reward_dp / (time_dp))
        result['final_time'] = float(time_dp)
        result['playing_br'] = ddp_play_rates
        result['rebuff'] = float(buffer_dp)

        with open(path + "/DDP.json", 'w') as writer:
            json.dump(result, writer)

        print("DDP finished work for sample {0}".format(sample))

    # print("Bola: r = {0}, time={1}, b={2}, r_0={3}".format(reward_bola/ time_bola,time_bola, buffer_bola, reward_bola))
    # print("DPP: r = {0}, time={1}, b={2}, r_0={3}".format(reward_dp/ time_dp,time_dp, buffer_dp, reward_dp))

    # plt.plot(max_bands, bola_performance, label="Bola360")
    # plt.plot(max_bands, ddpO_performance, label="DDP-Online")
    # plt.legend()
    # plt.xlabel("Bandwidth's average")
    # plt.title("Objective values of Bola360 vs DDP-Online")
    # plt.savefig("bandwidth_change.png", dpi=600)

__run_naive__ = True
#############################
#############################
###       Naive           ###
#############################
#############################

if __run_naive__:

    sample_paths = get_sample_paths()
    for [path, sample] in sample_paths:
        print("Naive starts work for sample {0}".format(sample))
        meta = read_meta(path + "/meta.json")
        N = meta['N']
        D = meta['D']
        buffer_size = meta['b_max']
        delta = meta['delta']
        gamma = meta['gamma']
        t_0 = meta['t_0']
        wait_time = meta['wait']
        sizes = np.array(meta['sizes'])
        bandwidth_error = meta['b_error']
        v_coeff = meta['V']
        actual_movement = meta['view']

        video = Video(N, delta, D, values, sizes, buffer_size)
        bandwidth = Bandwidth(data_path, error_rate=bandwidth_error)
        nav_graph_path = "dataset/headmovement/navG1.json"

        headMovements = HeadMoves(N, D, path=nav_graph_path)
        headMovements.set_actual_views(actual_movement)

        naive_solutions = []
        buffer_levels_naive = []
        available_content = [-1 for _ in range(N)]
        buffer_array_naive = [0 for _ in range(N)]
        time_levels_naive = []

        naive_play_rates = []

        time_naive = 0
        buffer_naive = 0  # unit: time
        reward_naive = 0
        rebuffer_naive = 0
        total_reward_naive = 0

        __tile_to_download__ = 1
        naive_alg = Naive(video, buffer_size, __tile_to_download__)
        # naive algorithm
        n = 0

        while True:
            if n >= N:
                break
            time_levels_naive.append(time_naive)
            buffer_levels_naive.append(buffer_naive)
            probs = headMovements.get_pose_probs(n)

            bandwidth_capacity = bandwidth.get_thr(time_naive)
            naive_action = naive_alg.get_action(probs, bandwidth_capacity)
            naive_solutions.append(naive_action)

            if np.sum(naive_action) > 0:
                total_size, download_n = get_total_size(naive_action, video)
                download_time_naive = bandwidth.download_time(total_size, time_naive)
                buffer_array_naive[n] = download_n
                naive_alg.take_action(naive_action, n, time_naive)
                naive_play_rates.append(
                    (time_naive + download_time_naive, convert_action_to_rates(naive_action, sizes, delta)))

            else:
                download_n = 0
                download_time_naive = wait_time
                naive_alg.take_action(naive_action, n, time_naive)

            time_naive += max(download_time_naive,
                              (download_n - buffer_size) * delta + buffer_naive - download_time_naive)

            if np.sum(naive_action) > 0:
                available_content[n - 1] = time_naive

            reward_naive, buffer_naive, reb, buffer_array_naive = calc_reward(naive_action, n, delta, probs,
                                                                              video,
                                                                              gamma, buffer_array_naive,
                                                                              available_content, time_naive,
                                                                              rebuffer_naive)

            if np.sum(naive_action) > 0:
                n += 1

            naive_alg.set_buffer(buffer_naive / delta)
            rebuffer_naive += reb
            total_reward_naive += reward_naive
        # print("Bola performance: {0}".format(total_reward_bola / (time_bola + t_0)))

        result = {}
        result['solution'] = naive_solutions
        result['time'] = time_levels_naive
        result['buffer'] = [int(x) for x in buffer_levels_naive]
        result['reward'] = float(total_reward_naive / (time_naive + t_0))
        result['final_time'] = float(time_naive + t_0)
        result['playing_br'] = naive_play_rates
        result['rebuff'] = float(rebuffer_naive)

        with open(path + "/naive_{0}.json".format(__tile_to_download__), 'w') as writer:
            json.dump(result, writer)

        print("Naive {1} finished work for sample {0}".format(sample, __tile_to_download__))

__plot_bandwidth__ = False
if __plot_bandwidth__:
    plot_bandwidth(bandwidth.throuput_changes, bola_play_rates, 0.1, max(time_bola, delta * N), sizes / delta, v_coeff,
                   D)
