from Video import *

import numpy as np

class Bola3d:
    def __init__(self, video, gamma, v_coeff):
        self.video = video
        #self.head_move = head_move
        self.gamma = gamma
        self.buffer = 0
        self.D = video.D
        self.M = len(video.values)
        self.V = v_coeff * (video.buffer_size - self.D) / (video.values[-1] + gamma * video.delta)
        self.all_sols = self.get_all_solutions(self.D)
        self.downloaded_segments = [0 for _ in range(self.video.N)]
        self.last_finished_segments = -1

    def get_action(self, probs):
        all_sols = self.all_sols
        solution = None
        max_rho = -1
        for sol in all_sols:
            rho = self.calc_rho(sol, probs)
            if rho > max_rho:
                max_rho = rho
                solution = sol
        return solution

    def take_action(self, solution,n ,time):
        finished_segments = int(time / self.video.delta)
        for i in range(self.last_finished_segments, min(finished_segments, n + 1)):
            if i >= 0:
                self.buffer -= self.downloaded_segments[i]
        self.buffer = max(self.buffer, 0)
        number_of_downloaded_segments = 0
        for v in solution:
            if v > 0:
                number_of_downloaded_segments += 1
        self.buffer += number_of_downloaded_segments
        self.downloaded_segments[n] = number_of_downloaded_segments


    def calc_rho(self, solution, probs):
        if np.sum(solution) == 0:
            return 0
        rho = 0
        rho2 = 0
        for d in range(self.D):
            m = solution[d]
            v = self.video.values[m]
            rho += probs[d] * (self.V * (v + self.gamma * self.video.delta) - self.buffer)
            rho2 += self.video.sizes[m]
        return rho / rho2





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



