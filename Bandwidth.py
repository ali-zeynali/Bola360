from Video import *
import numpy as np
class Bandwidth:
    def __init__(self, min_val, max_val, delta, error_rate):
        self.min_val = min_val
        self.max_val = max_val
        self.bandwidth = self.constant_average
        self.delta = delta
        self.error_rate = error_rate

    def integral_of_bandwidth(self, begin, end):
        n = int((end - begin) / self.delta)
        result = 0
        for i in range(n):
            t1 = begin + i * self.delta
            t2 = t1 + self.delta
            result += self.delta * ( self.bandwidth(t1) + self.bandwidth(t2) ) / 2
        return result

    def get_finish_time(self, size, start):
        remaining = size
        time = start
        while remaining > 0:
            remaining -= self.bandwidth(time) * self.delta
            time += self.delta
        return time


    def constant_average(self, t):
        return (self.min_val + self.max_val) / 2



    def expected_download_time(self, solution, start_time, video):
        down_time = -1
        while down_time <= 0:
            dt = self.download_time(solution, start_time, video)
            rnd = np.random.random() * 2 * self.error_rate
            down_time = dt * (1 - self.error_rate + rnd) * dt
        return down_time


    def download_time(self, solution, start_time, video):
        size_of_download = 0
        for m in solution:
            size_of_download += video.sizes[m]
        return self.get_finish_time(size_of_download, start_time) - start_time
