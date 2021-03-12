
class HeadMoves:
    def __init__(self, N, D):
        self.N = N
        self.D = D
        self.all_probs = [[0 for _ in range(self.D)] for __ in range(self.N)]


    def get_all_probs(self):
        #TODO
        return self.all_probs

    def get_probs(self, n):
        #TODO
        return self.all_probs[n]
