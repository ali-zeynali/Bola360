from Video import *
from Bandwidth import *
class DDPOnline:
    def __init__(self, video, buffer_size, bandwidth,gamma, t_0 = 1):
        self.video = video
        self.buffer_size = buffer_size * self.video.delta
        self.D = video.D
        self.bandwidth = bandwidth
        self.t_0 = t_0
        self.Nb_max = self.buffer_size / (min(video.delta, t_0) / 10)
        self.Nt_max = int(self.video.sizes[-1] / (self.bandwidth.min_val * t_0))
        self.rewards = [[] for _ in range(self.Nt_max)]
        for t in range(self.Nt_max):
            self.rewards[t] = []
            for b in range(self.Nb_max):
                self.rewards[t].append(float('-inf'))


        self.solutions = None
        self.Time = float('inf')
        self.M = len(video.values)
        self.gamma = gamma


    def get_all_solutions(self, D):
        if D == 0:
            return [[]]
        sub_sol = self.get_all_solutions(D - 1)
        solution = []
        for m in range(self.M):
            for v in sub_sol:
                new_v = v.copy()
                new_v.append(m)
                solution.append(new_v)
        return solution

    def get_action(self, probs, t, b, r0):
        all_sols = self.get_all_solutions(self.D)
        for m in all_sols:
            x = self.bandwidth.expected_download_time(m, t)
            x0 = int(x / self.t_0) * self.t_0
            xp = max(x0, b + self.D * self.video.delta - self.buffer_size)
            y = max(xp - b, 0)
            tp = t + xp
            bp = b + xp + y + self.D * self.video.delta
            expected_vals = 0
            for i in range(self.D):
                expected_vals += probs[i] * self.video.values[m[i]]

            rp = r0 +  self.gamma * (self.video.delta - y) + expected_vals
            if rp / tp > self.rewards[tp][bp] / self.Time:
                self.solutions = m
                self.Time = t

            self.rewards[t][b] = max(self.rewards[tp][bp], rp)

    def get_optimal_solutions(self):
        return self.solutions












