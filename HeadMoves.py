import json

import numpy as np


class HeadMoves:
    def __init__(self, N, D, path=None):
        self.N = N
        self.D = D
        self.all_probs = []
        self.path = path
        self.actual_views = []
        self.initialize_probs()

    def nav_graph(self, path):
        D = self.D
        all_probs = []
        if path is not None:
            with open(path) as file:
                raw_navigation_graph = json.load(file)

            previous_views = [self.binary_view("0001")]
            for segment_entry in raw_navigation_graph:
                f1_graph = np.zeros((D, D))
                views = [self.binary_view(hex_view) for hex_view in segment_entry['views']]

                transitions = np.array(segment_entry['transitions'], dtype=float)
                (tr_h, tr_w) = transitions.shape
                for i in range(tr_h):  # over current views
                    view = views[i]
                    for j in range(tr_w):  # over previous views
                        if transitions[i][j] > 0:
                            pr_view = previous_views[j]
                            for current_tile in range(D):  # for each tile inside current tiles
                                if view[current_tile] == "1":
                                    for pr_tile in range(D):  # for each tile inside previous tiles
                                        if pr_view[pr_tile] == "1":
                                            f1_graph[pr_tile][current_tile] += transitions[i][j]
                previous_views = views
                for i in range(D):
                    count = np.sum(f1_graph[i])
                    if count == 0:
                        f1_graph[i] = np.ones(D) / D
                    else:
                        f1_graph[i] = f1_graph[i] / count
                all_probs.append(f1_graph)
        self.all_probs = all_probs

    def binary_view(self, hex_view):
        zeros = 0
        for c in hex_view:
            if c == "0":
                zeros += 1
            else:
                break

        b_view = bin(int(hex_view, 16))[2:].zfill(8)
        return "0000" * zeros + str(b_view)

    def initialize_probs(self):
        self.nav_graph(self.path)

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

    def set_actual_views(self, actual_views):
        self.actual_views = actual_views

    def get_sample_actual_view(self):
        act_views = []
        for n in range(self.N):
            if n == 0:
                act_view = np.random.choice(range(self.D), 1)[0]
                act_views.append(act_view)
            else:
                last_view = act_views[-1]
                probs = self.all_probs[n][last_view]
                act_view = np.random.choice(range(self.D), 1, p=probs)[0]
                act_views.append(act_view)
        return act_views

    def get_all_probs(self):
        all_probs = []
        n = 0
        for tile in self.actual_views:
            all_probs.append(self.all_probs[n][tile])
            n += 1
        return all_probs

    def get_pose_probs(self, segment_number):
        current_tile = 0 if segment_number == 0 else self.actual_views[segment_number-1]
        return self.all_probs[segment_number][current_tile]
