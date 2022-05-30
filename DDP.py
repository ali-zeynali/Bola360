from Bandwidth import *


class DDP:
    def __init__(self, video, buffer_size, bandwidth, gamma, t_0=1):
        self.video = video
        self.buffer_size = buffer_size * self.video.delta
        # self.head_moves = head_moves
        self.D = video.D
        self.bandwidth = bandwidth
        self.t_0 = t_0
        self.Nb_max = 50
        self.b_0 = buffer_size * video.delta / self.Nb_max
        self.N = video.N
        # self.Nt_max = int(self.video.sizes[-1] * self.N * self.D / (self.bandwidth.min_val * t_0))
        self.Nt_max = self.N * video.delta * 2 / t_0
        self.Nt_max = int(self.Nt_max)
        self.rewards = [[] for _ in range(self.N + 1)]
        for n in range(self.N + 1):
            self.rewards[n] = [[] for _ in range(self.Nt_max)]
            for t in range(self.Nt_max):
                self.rewards[n][t] = []
                for b in range(self.Nb_max):
                    self.rewards[n][t].append(float('-inf'))
                    if n == 0:
                        self.rewards[n][t][b] = 0

        self.solutions = [None for _ in range(self.N)]
        self.Time = [self.t_0 * self.Nt_max for _ in range(self.N)]
        self.M = len(video.values)
        self.gamma = gamma
        self.all_sols = self.get_all_solutions(self.D)

    def get_all_solutions(self, D):
        # fixed
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

    def get_total_size(self, sizes):
        # fixed
        total_size = 0
        n = 0
        for m in sizes:
            total_size += self.video.sizes[m]
            if m > 0:
                n += 1
        return total_size, n

    def train(self, all_probs):
        for n in range(1, self.N + 1):
            print("Calculating Optimal solution for: n = {1} / {0}".format(self.N, n))
            for tp in range(self.Nt_max):
                for bp in range(self.Nb_max):
                    if self.rewards[n - 1][tp][bp] > float('-inf'):
                        all_sols = self.all_sols
                        for m in all_sols:
                            if np.sum(m) == 0:
                                continue
                            total_size, download_n = self.get_total_size(m)
                            x = self.bandwidth.download_time(total_size, tp * self.t_0)
                            x0 = int(x / self.t_0) * self.t_0
                            xp = max(x0,
                                     bp * self.b_0 + download_n * self.video.delta - self.buffer_size * self.video.delta)
                            y = max(xp - bp * self.b_0, 0)
                            t = tp * self.t_0 + xp
                            b = max(bp * self.b_0 - xp + y + download_n * self.video.delta, 0)
                            if int(b / self.b_0) >= self.Nb_max:
                                continue
                            if int(t / self.t_0) >= self.Nt_max:
                                continue

                            expected_vals = 0
                            for i in range(self.D):
                                expected_vals += all_probs[n-1][i] * self.video.values[m[i]]

                            rp = self.rewards[n - 1][tp][bp] + self.gamma * (self.video.delta) + expected_vals
                            print("n: {0}, tp:{1}, bp:{2}, t:{3}, b:{4} N_t:{5}, N_b:{6}".format(n, tp, bp,
                                                                                                 int(t / self.t_0),
                                                                                                 int(b / self.b_0),
                                                                                                 self.Nt_max,
                                                                                                 self.Nb_max))
                            new_val = rp / t
                            old_val = self.rewards[n][int(t / self.t_0)][int(b / self.b_0)] / self.Time[n - 1]
                            if new_val > old_val:
                                self.solutions[n - 1] = m
                                self.Time[n - 1] = t

                            self.rewards[n][int(t / self.t_0)][int(b / self.b_0)] = max(
                                self.rewards[n][int(t / self.t_0)][int(b / self.b_0)], rp)

    def get_optimal_solutions(self):
        return self.solutions

    def get_optimal_reward(self):
        max_rev = -1
        for t in range(self.Nt_max):
            for b in range(self.Nb_max):
                if self.rewards[self.N][t][b] / (t * self.t_0 + self.t_0) > max_rev:
                    max_rev = self.rewards[self.N][t][b] / (t * self.t_0 + self.t_0)
        return max_rev
