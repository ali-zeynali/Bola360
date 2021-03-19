import numpy as np
class HeadMoves:
    def __init__(self, N, D):
        self.N = N
        self.D = D
        self.all_probs = []
        self.initialize_probs()


    def initialize_probs(self):
        probs = []
        for n in range(self.N):
            ps = []
            for d in range(self.D):
                ps.append(np.random.random())
            ps = np.array(ps) / np.sum(ps)
            probs.append(ps)
        self.all_probs = probs


    def get_all_probs(self):
        #TODO
        return self.all_probs

    def get_probs(self, n):
        #TODO
        return self.all_probs[n]
