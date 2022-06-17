import glob
import os
import shutil

from Bola3d import *
from DDP import *
from DDPOnline import *
from Evaluate import *
from HeadMoves import *
from Naive import *
from Video import *


# TODO: rebuffering must be applied based on the actual head movement

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
    reb = None
    for i in range(segment):
        if available_time[i] > 0:
            last_available = available_time[i]

        if buffer_array[i] > 0:
            reb = 0
            break

    if reb is None:
        # if time > last_available + delta:
        reb = time - last_available - delta
        x = 1

    if reb < 0:
        print("OMG")
    # y = max(download_time - buffer, 0) #TODO, this line has a bug
    # buffer = max(buffer - download_time, 0) + ddp_n * video.delta
    # buffer = buffer - download_time + y + ddp_n * video.delta  # TODO, this line has a bug
    buffer = np.sum(buffer_array) * delta
    expected_vals_dp = 0
    expected_smooth = 0
    for i in range(D):
        if solution[i] > 0:
            expected_vals_dp += probs[i] * video.values[solution[i]]
            expected_smooth += probs[i] * delta
    r0 = expected_vals_dp + expected_smooth * gamma
    if ddp_n > 0:
        return r0, expected_vals_dp, buffer, reb, buffer_array
    else:
        return 0, 0, buffer, reb, buffer_array


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


def plot_algorithm_changes(changes, algorithms, sizes, delta, dt, bitrates, path, DDP=False, __start_time__=0):
    global figure_number
    figure_number += 1
    fig = plt.figure(figure_number)

    final_time = 0
    for name in algorithms:
        if not DDP and name == "DP-Online":
            continue
        algorithm = algorithms[name]
        solutions = algorithm['solution']
        times = algorithm['time']

        if times[-1] > final_time:
            final_time = times[-1]
        play_rate = []
        time_stamp = []
        previous_rate = -1
        for i in range(len(solutions)):
            size = 0
            for a in solutions[i]:
                size += sizes[a]
            if previous_rate >= 0:
                time_stamp.append(times[i])
                play_rate.append(previous_rate)

            previous_rate = size / delta
            time_stamp.append(times[i])
            play_rate.append(size / delta)

        plt.plot(time_stamp, play_rate, label=name, linewidth=1.5)

    x = []
    y = []
    t = __start_time__
    indx = 0
    changes = changes.copy()
    flag = False
    while t > changes[indx][0]:
        indx += 1
        flag = True
    if flag:
        indx -= 1
    while t < final_time:
        while changes[indx][0] < t:
            indx += 1
        x.append(t)
        y.append(changes[indx][1])
        t += dt
    plt.plot(x, y, label="Network", linewidth=1)

    for br in bitrates:
        plt.plot([__start_time__, final_time], [br, br], label='_nolegend_', linestyle='dashed', color="gray",
                 linewidth=0.7)
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Bitrate (Mbps)")
    plt.savefig(path, dpi=600, bbox_inches='tight')


def plot_algorithm_changes_avg(changes, algorithms, sizes, delta, dt, bitrates, path, DDP=False, __start_time__=0):
    global figure_number
    figure_number += 1
    fig = plt.figure(figure_number)

    final_time = 0
    for name in algorithms:
        if not DDP and name == "DP-Online":
            continue
        algorithm = algorithms[name]
        solutions = algorithm['solution']
        times = algorithm['time']

        if times[-1] > final_time:
            final_time = times[-1]
        play_rate = []
        time_stamp = []
        previous_rate = -1
        for i in range(len(solutions)):
            size = []
            for a in solutions[i]:
                if a > 0:
                    size.append(sizes[a])
            if previous_rate >= 0:
                time_stamp.append(times[i])
                play_rate.append(previous_rate)
            size = np.average(size) if len(size) > 0 else 0
            previous_rate = size / delta
            time_stamp.append(times[i])
            play_rate.append(size / delta)

        plt.plot(time_stamp, play_rate, label=name, linewidth=1.5)

    x = []
    y = []
    t = __start_time__
    indx = 0
    changes = changes.copy()
    flag = False
    while t > changes[indx][0]:
        indx += 1
        flag = True
    if flag:
        indx -= 1
    changes = changes.copy()
    while t < final_time:
        while changes[indx][0] < t:
            indx += 1
        x.append(t)
        y.append(changes[indx][1])
        t += dt
    plt.plot(x, y, label="Network", linewidth=1)

    for br in bitrates:
        plt.plot([__start_time__, final_time], [br, br], label='_nolegend_', linestyle='dashed', color="gray",
                 linewidth=0.7)
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Bitrate (Mbps)")
    plt.savefig(path, dpi=600, bbox_inches='tight')


def plot_bandwith(changes, dt, final_time, bitrates, name):
    global figure_number
    figure_number += 1
    fig = plt.figure(figure_number)
    x = []
    y = []
    t = 0
    indx = 0
    changes = changes.copy()
    while t < final_time:
        while changes[indx][0] < t:
            indx += 1
        x.append(t)
        y.append(changes[indx][1])
        t += dt
    plt.plot(x, y, label="Network", linewidth=1.5)

    for br in bitrates:
        plt.plot([0, final_time], [br, br], label='_nolegend_', linestyle='dashed', color="gray", linewidth=0.7)
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Bitrate (Mbps)")
    plt.savefig("figures/{0}.png".format(name), dpi=600, bbox_inches='tight')


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


def plot_values(values, gamma, delta, path):
    global figure_number
    figure_number += 1
    fig = plt.figure(figure_number)
    m = len(values)
    plt.plot([i + 1 for i in range(m)], values, label="values", linewidth=1.5)
    plt.plot([1, m], [gamma * delta, gamma * delta], label="$\gamma * \delta$", linewidth=1.5)

    plt.xlabel("Bitrate index")
    plt.legend()

    plt.savefig(path, dpi=600, bbox_inches='tight')


def run_bola_experiment(__print__=True, __start_time__=0):
    if __print__:
        print("------> Running Bola360 algorithm")
    sample_paths = get_sample_paths()
    for [path, sample] in sample_paths:
        if __print__:
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
        time_bola = __start_time__
        buffer_bola = 0  # unit: time
        reward_bola = 0
        rebuffer_bola = 0
        total_reward_bola = 0
        total_reward_val_bola = 0

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
            # print("Action for n:{0}, \t{1}".format(n, bola_action))
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
                available_content[n] = time_bola

            reward_bola, reward_val_bola, buffer_bola, reb, buffer_array_bola = calc_reward(bola_action, n, delta,
                                                                                            probs, video,
                                                                                            gamma,
                                                                                            buffer_array_bola,
                                                                                            available_content,
                                                                                            time_bola,
                                                                                            rebuffer_bola)
            if np.sum(bola_action) > 0:
                n += 1

            # if n >= N:
            #     time_levels_bola.append(time_bola)
            #     buffer_levels_bola.append(buffer_bola)

            bola3d.set_buffer(buffer_bola / delta)

            rebuffer_bola += reb
            total_reward_bola += reward_bola
            total_reward_val_bola += reward_val_bola
            # print("Buffer: {0}".format(buffer_bola / delta))
            # print(("Buffer-bola: {0}".format(bola3d.buffer)))
        # print("Bola performance: {0}".format(total_reward_bola / (time_bola + t_0)))

        result = {}
        result['solution'] = bola_solutions
        result['time'] = time_levels_bola
        result['buffer'] = [int(x) for x in buffer_levels_bola]
        result['reward'] = float(total_reward_bola / (time_bola))
        result['final_time'] = float(time_bola)
        result['playing_br'] = bola_play_rates
        result['rebuff'] = float(rebuffer_bola)
        result['reward_val'] = total_reward_val_bola / time_bola

        with open(path + "/bola.json", 'w') as writer:
            json.dump(result, writer)
        if __print__:
            print("Bola finished work for sample {0}".format(sample))


def run_ddp_experiment(__start_time__=0):
    print("------> Running DP-Online algorithm")
    sample_paths = get_sample_paths()
    for [path, sample] in sample_paths:
        print("DDP starts work for sample {0}".format(sample))
        meta = read_meta(path + "/meta.json")
        N = meta['N']
        # N = 20  # TODO: remove this
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

        time_dp = __start_time__
        buffer_dp = 0
        reward_dp = 0
        rebuffer_dp = 0
        ddp_solutions = []
        buffer_levels_ddp = []
        available_content = [-1 for _ in range(N)]
        buffer_array_ddp = [0 for _ in range(N)]
        time_levels_ddp = []
        ddpO_performance = []
        total_reward_val_ddp = 0

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
                available_content[n] = time_dp

            reward_ddp, reward_val_ddp, buffer_dp, reb, buffer_array_ddp = calc_reward(ddp_action, n, delta, probs,
                                                                                       video,
                                                                                       gamma, buffer_array_ddp,
                                                                                       available_content,
                                                                                       time_dp, rebuffer_dp)

            if np.sum(ddp_action) > 0:
                n += 1

            ddp_online.set_buffer(buffer_dp / delta)
            rebuffer_dp += reb
            reward_dp += reward_ddp
            total_reward_val_ddp += reward_val_ddp

        result = {}
        result['solution'] = ddp_solutions
        result['time'] = time_levels_ddp
        result['buffer'] = [int(x) for x in buffer_levels_ddp]
        result['reward'] = float(reward_dp / (time_dp))
        result['final_time'] = float(time_dp)
        result['playing_br'] = ddp_play_rates
        result['rebuff'] = float(rebuffer_dp)
        result['reward_val'] = total_reward_val_ddp / time_dp

        with open(path + "/DDP.json", 'w') as writer:
            json.dump(result, writer)

        print("DDP finished work for sample {0}".format(sample))


def run_naive_experiment(__tile_to_download__, __start_time__=0):
    # __tile_to_download__ = 2
    print("------> Running Naive {0} algorithm".format(__tile_to_download__))
    sample_paths = get_sample_paths()
    for [path, sample] in sample_paths:
        print("Naive starts work for sample {0}".format(sample))
        meta = read_meta(path + "/meta.json")
        N = meta['N']
        # N = 20  # TODO: remove this
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

        time_naive = __start_time__
        buffer_naive = 0  # unit: time
        reward_naive = 0
        rebuffer_naive = 0
        total_reward_naive = 0
        total_reward_val_naive = 0

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
                available_content[n] = time_naive

            reward_naive, reward_val_naive, buffer_naive, reb, buffer_array_naive = calc_reward(naive_action, n, delta,
                                                                                                probs,
                                                                                                video,
                                                                                                gamma,
                                                                                                buffer_array_naive,
                                                                                                available_content,
                                                                                                time_naive,
                                                                                                rebuffer_naive)

            if np.sum(naive_action) > 0:
                n += 1

            naive_alg.set_buffer(buffer_naive / delta)
            rebuffer_naive += reb
            total_reward_naive += reward_naive
            total_reward_val_naive += reward_val_naive
        # print("Bola performance: {0}".format(total_reward_bola / (time_bola + t_0)))

        result = {}
        result['solution'] = naive_solutions
        result['time'] = time_levels_naive
        result['buffer'] = [int(x) for x in buffer_levels_naive]
        result['reward'] = float(total_reward_naive / (time_naive + t_0))
        result['final_time'] = float(time_naive + t_0)
        result['playing_br'] = naive_play_rates
        result['rebuff'] = float(rebuffer_naive)
        result['reward_val'] = total_reward_val_naive / time_naive

        with open(path + "/naive_{0}.json".format(__tile_to_download__), 'w') as writer:
            json.dump(result, writer)

        print("Naive {1} finished work for sample {0}".format(sample, __tile_to_download__))


def tune_bola(N, D, buffer_size, delta, gamma, t_0, wait_time, sizes, bandwidth_error, actual_movement,
              __start_time__=0, __learning_rate__=0.01):
    V_ranges = np.arange(0.6, 1, __learning_rate__)

    performances = {}
    for v in V_ranges:
        print("Checking V-coeff value: {0}".format(v))
        samples = 4
        for i in range(samples):
            sample_path = "results/sample_{0}/".format(i)
            save_meta(N, D, buffer_size, delta, gamma, t_0, wait_time, sizes, bandwidth_error, v, actual_movement,
                      path=sample_path + "meta.json")
        run_bola_experiment(__print__=False, __start_time__=__start_time__)

        bola_perf, _, __, ___, ____ = compare_rewards(number_of_samples=4, __print__=False)
        performances[v] = bola_perf
    max_v = 0
    max_perf = 0
    for v in performances:
        perf = np.average(performances[v])
        if perf > max_perf:
            max_perf = perf
            max_v = v
    print("Best v was: {0} with performance: {1}".format(max_v, max_perf))
    return max_v


# ratio = 2
N = 40
D = 6

buffer_size = 20  # number of segments fit into buffer, unit: integer
delta = 5
gamma = 0.5
t_0 = delta / 10
wait_time = t_0

data_path = "dataset/bandwidth/4G/4G_BWData.json"
# data_path = "fix_bandwidth.json"

bandwidth_error = 0.10
sizes = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]) * delta / 2
sizes = np.array([0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12]) * delta

sizes = np.array([0, 1, 2, 4, 6, 8]) * delta

# sizes = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) * delta
M = len(sizes) - 1
# v_coeff = buffer_size / (buffer_size - D)
# v_coeff = 0.98
__start_time__ = 0
# values = np.array([i for i in range(M)]) * delta * 50/ 2
# for i in range(len(values)):
#     values[i] = np.power(values[i], 2)
values = np.array([0 if x == 0 else np.log(x / sizes[1]) for x in sizes])
# values = np.array([0 if x == 0 else x * np.log(x / sizes[1]) for x in sizes]) / 14
# v_coeff = 0.7 * (values[-1] + gamma * delta) / (buffer_size - D)  # TODO: update/remove this

video = Video(N, delta, D, values, sizes, buffer_size)
bandwidth = Bandwidth(data_path, error_rate=bandwidth_error)
nav_graph_path = "dataset/headmovement/navG1.json"

headMovements = HeadMoves(N, D, path=nav_graph_path)

actual_movement = headMovements.get_sample_actual_view()
v_coeff = 0.9
# v_coeff = tune_bola(N, D, buffer_size, delta, gamma, t_0, wait_time, sizes, bandwidth_error, actual_movement,
#                     __start_time__=__start_time__, __learning_rate__=0.02)

__save_meta__ = False
if __save_meta__:
    samples = 1
    for i in range(samples):
        actual_movement = headMovements.get_sample_actual_view()
        headMovements.set_actual_views(actual_movement)
        sample_path = "results/sample_{0}/".format(i)
        # try:
        #     shutil.rmtree(sample_path)
        # except Exception:
        #     pass
        #
        # os.mkdir(sample_path)

        save_meta(N, D, buffer_size, delta, gamma, t_0, wait_time, sizes, bandwidth_error, v_coeff, actual_movement,
                  path=sample_path + "meta.json")
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

# run_bola_experiment(__start_time__=__start_time__)
# run_ddp_experiment(__start_time__=__start_time__)
# run_naive_experiment(1, __start_time__=__start_time__)
# run_naive_experiment(3, __start_time__=__start_time__)
# run_naive_experiment(6, __start_time__=__start_time__)

__plot_bandwidth__ = True
if __plot_bandwidth__:
    figure_number = 0
    sample_paths = get_sample_paths()
    for [path, sample] in sample_paths:
        print("Plotting for sample: {0}".format(sample))
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

        bola = read_meta(path + "/bola.json")
        ddp = read_meta(path + "/DDP.json")
        naive_1 = read_meta(path + "/naive_1.json")
        naive_h = read_meta(path + "/naive_3.json")
        naive_f = read_meta(path + "/naive_6.json")

        algorithms = {}
        algorithms['Bola360'] = bola
        algorithms['DP-Online'] = ddp
        algorithms['Naive1'] = naive_1
        algorithms['Naive-h'] = naive_h
        algorithms['Naive-f'] = naive_f

        plot_algorithm_changes(bandwidth.throuput_changes, algorithms, sizes, delta, 0.1, sizes / delta,
                               path + "/algorithms_V_B{0}_S{1}_g{2}.png".format(buffer_size, __start_time__, gamma,
                                                                                v_coeff),
                               DDP=False, __start_time__=__start_time__)
        plot_algorithm_changes_avg(bandwidth.throuput_changes, algorithms, sizes, delta, 0.1, sizes / delta,
                                   path + "/algorithms_V_B{0}_S{1}_g{2}_avg.png".format(buffer_size, __start_time__,
                                                                                        gamma,
                                                                                        v_coeff), DDP=False,
                                   __start_time__=__start_time__)

        compare_buffers(path + "/buffers_{0}_S{1}_g{2}.png".format(buffer_size, __start_time__, gamma), sample, delta,
                        DDP=False,
                        __start_time__=__start_time__)
        plot_values(values, gamma, delta, path + "/values_g{0}.png".format(gamma))

        real_evaluation(path + "/play_bitrate_B{0}_S{1}_g{2}.png".format(buffer_size, __start_time__, gamma), sample,
                        delta,values, DDP=False, __start_time__=__start_time__, __print__=True, N1=True, Nf=True, Nh=False)
    bandwidth = Bandwidth(data_path, error_rate=bandwidth_error)
    plot_bandwith(bandwidth.throuput_changes, 0.1, 1000, sizes / delta, "Bandwidth_4G_BW")

    compare_rewards(number_of_samples=1)
    compare_rewards_val(number_of_samples=1)
    compare_rebuff(number_of_samples=1)
