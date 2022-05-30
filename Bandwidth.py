import json

import numpy as np


class Bandwidth:
    def __init__(self, path_dir, error_rate):

        self.min_val = float('inf')
        self.max_val = float('-inf')
        self.delta = 0.01
        self.error_rate = error_rate
        self.current_tp = np.average([0, 0])
        self.throuput_changes = []  # (time, val)
        self.last_updated_time = 0
        # self.dt = 0.05
        self.read_dataset(path_dir)
        self.calculated_download_time = {}

    def read_dataset(self, path_dir):
        with open(path_dir, 'r') as reader:
            dataset = json.load(reader)
        time = 0
        for tuple in dataset['results']:
            dlrate = tuple['dlRate'] / 1000
            self.throuput_changes.append((time, dlrate))
            time += tuple['dlDuration']
            if self.min_val > dlrate:
                self.min_val = dlrate
            if self.max_val < dlrate:
                self.min_val = dlrate
        self.current_tp = self.throuput_changes[0]
        # pass #TODO: read directory and initialize values

    def get_thr(self, time):
        if time > self.last_updated_time:
            return self.throuput_changes[-1][1]
        for (t, thr) in self.throuput_changes:
            if time <= t:  # TODO: need to update this, thr must update after thrigger not before it
                return thr
        return self.throuput_changes[-1][1]

    def integral_of_bandwidth(self, begin, end):

        n = int((end - begin) / self.delta)
        result = 0
        for i in range(n):
            t1 = begin + i * self.delta
            t2 = t1 + self.delta
            result += self.delta * (self.get_thr(t1) + self.get_thr(t2)) / 2
        return result

    def get_finish_time(self, size, start):
        remaining = size
        time = start
        while remaining > 0:
            remaining -= self.get_thr(time) * self.delta
            time += self.delta
        return time

    def expected_download_time(self, segment_size, start_time):
        down_time = -1
        while down_time <= 0:
            dt = self.download_time(segment_size, start_time)
            rnd = np.random.random() * 2 * self.error_rate
            down_time = dt * (1 - self.error_rate + rnd)
        return down_time

    def download_time(self, total_size, start_time):
        if start_time in self.calculated_download_time:
            if total_size in self.calculated_download_time[start_time]:
                return self.calculated_download_time[start_time][total_size]
            else:
                calculated_time = self.get_finish_time(total_size, start_time) - start_time
                self.calculated_download_time[start_time][total_size] = calculated_time

                return calculated_time
        else:
            self.calculated_download_time[start_time] = {}

            calculated_time = self.get_finish_time(total_size, start_time) - start_time
            self.calculated_download_time[start_time][total_size] = calculated_time

            return calculated_time
