from Video import *
from Bandwidth import *
class DDP:
    def __init__(self, video, buffer_size, bandwidth,gamma, t_0 = 1):
        self.video = video
        self.buffer_size = buffer_size * self.video.delta
        #self.head_moves = head_moves
        self.D = video.D
        self.bandwidth = bandwidth
        self.t_0 = t_0
        self.Nb_max = self.buffer_size / (min(video.delta, t_0) / 10)
        self.N = video.N
        self.Nt_max = int(self.video.sizes[-1] / (self.bandwidth.min_val * t_0))
        self.rewards = [[] for _ in range(self.N)]
        for n in range(self.N + 1):
            self.rewards[n] = [[] for _ in range(self.Nt_max)]
            for t in range(self.Nt_max):
                self.rewards[n][t] = []
                for b in range(self.Nb_max):
                    self.rewards[n][t].append(float('-inf'))
                    if n == 0:
                        self.rewards[n][t][b] = 0

        self.solutions = [None for _ in range(self.N)]
        self.Time = [float('inf') for _ in range(self.N)]
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

    def train(self, all_probs):
        for n in range(1, self.N + 1):
            #probs = self.head_moves.get_probabilities()
            for tp in range(self.Nt_max):
                for bp in range(self.Nb_max):
                    if self.rewards[n][tp][bp] > float('-inf'):
                        all_sols = self.get_all_solutions(self.D)
                        for m in all_sols:
                            x = self.bandwidth.download_time(m, tp)
                            x0 = int(x / self.t_0) * self.t_0
                            xp = max(x0, bp + self.D *self.video.delta - self.buffer_size)
                            y = max(xp - bp, 0)
                            t = tp + xp
                            b = bp + xp + y + self.D * self.video.delta
                            expected_vals = 0
                            for i in range(self.D):
                                expected_vals += all_probs[n-1][i] * self.video.values[m[i]]

                            rp = self.rewards[n-1][tp][bp] + self.gamma * (self.video.delta - y) + expected_vals
                            if rp / t > self.rewards[n][t][b] / self.Time[n-1]:
                                self.solutions[n-1] = m
                                self.Time[n-1] = t

                            self.rewards[n][t][b] = max(self.rewards[n][t][b], rp)
    def get_optimal_solutions(self):
        return self.solutions
    def get_optimal_reward(self):
        max_rev = -1
        for t in range(self.Nt_max):
            for b in range(self.Nb_max):
                if self.rewards[self.N][t][b] / t > max_rev:
                    max_rev = self.rewards[self.N][t][b] / t
        return max_rev










