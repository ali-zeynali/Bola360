import numpy as np
from matplotlib import pyplot as plt
import json


def get_rho(solution, values, sizes, probs, V, D, gamma, delta, Q):
    if np.sum(solution) == 0:
        return 0
    rho = 0
    rho2 = 0
    for d in range(D):
        m = solution[d]
        v = values[m]
        a = 0
        if m > 0:
            a = 1
        rho += probs[d] * a * (V * (v + gamma * delta) - Q)
        rho2 += sizes[m] * a
    return rho / rho2


def get_all_solutions(D, M):
    if D == 0:
        return [[]]
    sub_sol = get_all_solutions(D - 1, M)
    solution = []
    for m in range(M):
        for v in sub_sol:
            new_v = v.copy()
            new_v.append(m)
            solution.append(new_v)
    return solution


def get_aggregate_bitrate(sol, sizes, delta):
    bitrates = np.array(sizes) / delta
    aggr_br = 0
    for i in range(len(sol)):
        aggr_br += bitrates[sol[i]]
    return aggr_br


def plot_decision_thresholds(video, buffer_max, gamma, probs, dq=0.05):
    values = video.values
    sizes = video.sizes

    buffer_vals = np.arange(0, buffer_max, dq)
    rho_vals = [[] for _ in range(len(buffer_vals))]
    all_sols = get_all_solutions(video.D, len(sizes))

    for Q, i in enumerate(buffer_vals):
        for sol in all_sols:
            rho = get_rho(sol, values, sizes, probs, video.V, video.D, gamma, video.delta, Q)
            rho_vals[i].append(rho)

    max_sols = []
    max_rhos = []
    for i in range(len(buffer_vals)):
        sol_max_indx = np.argmax(rho_vals[i])
        max_rhos.append(rho_vals[i][sol_max_indx])
        max_sols.append(all_sols[sol_max_indx])

    fig = plt.figure(int(np.random.random() * 10000))
    plt.plot(buffer_vals * video.delta, max_rhos)
    plt.xlabel("Buffer levels (s)")
    plt.