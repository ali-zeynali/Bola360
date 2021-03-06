from HeadMoves import *
from Bandwidth import *
from DDP import *
from DDPOnline import *
from Bola3d import *
from Video import *
import matplotlib.pyplot as plt

def calc_reward(solution, buffer, download_time, delta,  probs, video, gamma):
    ddp_n = 0
    for m in solution:
        if m > 0:
            ddp_n += 1
    y = max(download_time - buffer, 0)
    buffer = max(buffer - download_time, 0) + ddp_n * video.delta
    expected_vals_dp = 0
    for i in range(D):
        expected_vals_dp += probs[i] * video.values[solution[i]]
    r0 = gamma * (delta) + expected_vals_dp
    return r0, buffer





N = 20
D = 4
M = 3
buffer_size = 5
delta = 5
gamma = 2
t_0 = delta / 10
wait_time = t_0
bandwidth_error = 0.10
sizes = np.array([i for i in range(M)]) * 10
v_coeff = 1
values = np.array([i for i in range(M)]) * 10

min_band = 0.1
max_band = 10

video = Video(N, delta, D, values, sizes, buffer_size)
bandwidth = Bandwidth(min_band, max_band, delta=t_0 / 1000, error_rate=bandwidth_error)
headMovements = HeadMoves(N, D)

# bola3d = Bola3d(video, gamma, v_coeff)
# ddp = DDP(video,buffer_size, bandwidth, gamma, t_0)
# ddp_online = DDPOnline(video,buffer_size, bandwidth, gamma, t_0)






print("Calculating optimal offline")
# ddp.train(headMovements.get_all_probs())
# optimal_offline  = ddp.get_optimal_reward()
# print("Offline: ", optimal_offline)


bola_performance = []
ddpO_performance = []
max_bands = np.arange(5, 10, 0.1)

for max_band in max_bands:
    time_bola = 0
    buffer_bola = 0
    reward_bola = 0

    time_dp = 0
    buffer_dp = 0
    reward_dp = 0
    bandwidth = Bandwidth(min_band, max_band, delta=t_0 / 1000, error_rate=bandwidth_error)
    bola3d = Bola3d(video, gamma, v_coeff)
    ddp_online = DDPOnline(video, buffer_size, bandwidth, gamma, t_0)
    print(max_band)
    for n in range(N):
        print("n: ", n)
        probs = headMovements.get_probs(n)
        print("probs: ", probs)
        print("time: ", time_dp)
        #print("getting DP action")
        ddp_action = ddp_online.get_action(probs, time_dp, buffer_dp, reward_dp)
        print("n: " ,n ,"\tDDP-On action: ", ddp_action)
        download_time_dp = bandwidth.download_time(ddp_action, time_dp, video)

        time_dp += download_time_dp
        reward_dp, buffer_dp = calc_reward(ddp_action, buffer_dp, time_dp, delta, probs, video, gamma)
    n = 0
    while True:
        if n >= N:
            break
        print(n)
        probs = headMovements.get_probs(n)
        print("probs: ", probs)
        # print("getting bola action")
        bola_action = bola3d.get_action(probs)
        print("n: " ,n ,"\tBola action: ", bola_action)

        # print("getting DP action")

        if np.sum(bola_action) > 0:
            download_time_bola = bandwidth.download_time(bola_action, time_bola, video)
            bola3d.take_action(bola_action, n, time_bola)
            n += 1
        else:
            download_time_bola = wait_time
            bola3d.take_action(bola_action, n, time_bola)
        time_bola += download_time_bola

        reward_bola, buffer_bola = calc_reward(bola_action, buffer_bola, time_bola, delta, probs, video, gamma)

    bola_performance.append(reward_bola/ time_bola)
    ddpO_performance.append(reward_dp/ time_dp)
    print("Bola: r = {0}, time={1}, b={2}, r_0={3}".format(reward_bola/ time_bola,time_bola, buffer_bola, reward_bola))
    print("DPP: r = {0}, time={1}, b={2}, r_0={3}".format(reward_dp/ time_dp,time_dp, buffer_dp, reward_dp))


# print("Bola: r = {0}, time={1}, b={2}, r_0={3}".format(reward_bola/ time_bola,time_bola, buffer_bola, reward_bola))
# print("DPP: r = {0}, time={1}, b={2}, r_0={3}".format(reward_dp/ time_dp,time_dp, buffer_dp, reward_dp))


# plt.plot(max_bands, bola_performance, label="Bola360")
# plt.plot(max_bands, ddpO_performance, label="DDP-Online")
# plt.legend()
# plt.xlabel("Bandwidth's average")
# plt.title("Objective values of Bola360 vs DDP-Online")
# plt.savefig("bandwidth_change.png", dpi=600)




