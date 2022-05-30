import numpy as np
import json
class HeadMoves:
    def __init__(self, N, D, path=None):
        self.N = N
        self.D = D
        self.all_probs = []
        self.initialize_probs()
        if path is not None:
            with open(path) as file:
                raw_navigation_graph = json.load(file)

            self.graph = []
            for segment_entry in raw_navigation_graph:
                views = [int(x, 16) for x in segment_entry['views']]
                views_index = {}
                i = 0
                for v in views:
                    views_index[v] = i
                    i += 1
                transitions = np.array(segment_entry['transitions'], dtype=float)
                counts = np.sum(transitions, axis=0)  # sum columns
                transitions /= counts  # normalize probability to 1
                x = 1



    def initialize_probs(self):
        self.bin_probs()


    def random_probs(self):
        probs = []
        for n in range(self.N):
            ps = []
            for d in range(self.D):
                ps.append(np.random.random())
            ps = np.array(ps) / np.sum(ps)
            probs.append(ps)
        self.all_probs = probs

    def bin_probs(self):
        self.random_probs()

        new_probs = []
        for prob in self.all_probs:
            n = 0
            new_ps = []
            for p in prob:
                if p >= 1 / self.D:
                    n += 1
            for p in prob:
                if p >= 1 / self.D:
                    new_ps.append(1 / n)
                else:
                    new_ps.append(0)
            new_probs.append(new_ps)
        self.all_probs = new_probs

    def get_all_probs(self):
        return self.all_probs

    def head_movement(self, segment_number):
        #TODO
        return np.zeros(self.D)
