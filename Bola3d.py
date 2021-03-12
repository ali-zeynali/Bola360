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

    def get_action(self, probs):
        all_sols = self.get_all_solutions(self.D)
        solution = None
        max_rho = -1
        for sol in all_sols:
            rho = self.calc_rho(sol, probs)
            if rho > max_rho:
                max_rho = rho
                solution = sol
        return solution

    def take_action(self, solution):
        for v in solution:
            if v > 0:
                self.buffer += 1


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



