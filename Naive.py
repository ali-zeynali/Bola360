import numpy as np


class Naive:
    def __init__(self, video, buffer_capacity, tile_to_download):
        self.video = video
        self.tile_to_download = tile_to_download
        self.buffer = 0
        self.D = video.D
        self.M = len(video.values)
        self.downloaded_segments = [0 for _ in range(self.video.N)]
        self.last_finished_segments = -1
        self.buffer_capacity = buffer_capacity

    def get_action(self, probs, bandwidth_capacity):
        """
        :param probs: array size D, showing the probability of watching tiles
        :return: array size D, showing the selected bitrates of each tile
        """
        sorted_probs_indexes = len(probs) - 1 - np.argsort(probs)
        target_size = self.video.delta * bandwidth_capacity / self.tile_to_download
        sizes = self.video.sizes

        bitrate = 0
        for i in range(len(sizes) - 1):
            if sizes[i] <= target_size < sizes[i + 1]:
                bitrate = i
                break
        solution = [0 for _ in range(self.D)]
        for i in range(self.D):
            if sorted_probs_indexes[i] < self.tile_to_download:
                solution[i] = bitrate
        n = 0
        for a in solution:
            if a > 0:
                n += 1
        if self.buffer_capacity - self.buffer >= n:
            return solution
        else:
            return [0 for _ in range(self.D)]
    def set_buffer(self, buffer):
        self.buffer = buffer
    def take_action(self, solution, n, time):
        # finished_segments = int(time / self.video.delta)
        # finished_segments = min(finished_segments, n) # consider rebuff
        # for i in range(self.last_finished_segments, min(finished_segments, n + 1)):
        #     if i >= 0:
        #         self.buffer -= self.downloaded_segments[i]
        # self.last_finished_segments = finished_segments
        # self.buffer = max(self.buffer, 0)
        number_of_downloaded_segments = 0
        for v in solution:
            if v > 0:
                number_of_downloaded_segments += 1
        # self.buffer += number_of_downloaded_segments
        self.downloaded_segments[n] = number_of_downloaded_segments
