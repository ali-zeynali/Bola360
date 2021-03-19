from Video import *
from Bandwidth import *
class DDPOnline:
    def __init__(self, video, buffer_size, bandwidth,gamma, t_0 = 1):
        self.video = video
        self.buffer_size = buffer_size * self.video.delta
        self.D = video.D
        self.bandwidth = bandwidth
        self.t_0 = t_0
        self.Nb_max = 300
        self.b_0 = buffer_size * video.delta / self.Nb_max
        self.Nt_max = int(self.video.sizes[-1] * self.video.N * self.D / (self.bandwidth.min_val * t_0))
        self.rewards = [[] for _ in range(self.Nt_max)]
        for t in range(self.Nt_max):
            self.rewards[t] = []
            for b in range(self.Nb_max):
                self.rewards[t].append(float('-inf'))


        self.solutions = None
        self.Time = self.t_0 * self.Nt_max
        self.M = len(video.values)
        self.gamma = gamma
        self.all_sols = self.get_all_solutions(self.D)


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
        all_sols = self.all_sols
        for m in all_sols:
            if np.sum(m) == 0:
                continue
            x = self.bandwidth.expected_download_time(m, t, self.video)
            x0 = int(x / self.t_0) * self.t_0
            xp = max(x0, b + self.D * self.video.delta - self.buffer_size)
            y = max(xp - b, 0)
            tp = t + xp
            bp = b - xp + y + self.D * self.video.delta
            expected_vals = 0
            for i in range(self.D):
                expected_vals += probs[i] * self.video.values[m[i]]
            if int(bp / self.b_0) >= self.Nb_max:
                continue
            if int(tp / self.t_0) >= self.Nt_max:
                continue
            rp = r0 +  self.gamma * (self.video.delta) + expected_vals
            new_val = rp / tp
            old_val =  self.rewards[int(tp / self.t_0)][int(bp / self.b_0)] / self.Time
            if new_val > old_val:
                self.solutions = m
                self.Time = tp

            self.rewards[int(tp / self.t_0)][int(bp / self.b_0)] = max(self.rewards[int(tp / self.t_0)][int(bp / self.b_0)], rp)
        return self.solutions













