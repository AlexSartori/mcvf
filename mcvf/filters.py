from typing import List
import numpy as np


class Filter:
    def __init__(self):
        pass

    def filter_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        pass


class GaussianFilter(Filter):
    def __init__(self):
        self.kernel = np.array([
            [1,  4,  7,  4, 1],
            [4, 16, 26, 16, 4],
            [7, 26, 41, 26, 7],
            [4, 16, 26, 16, 4],
            [1,  4,  7,  4, 1]
        ])/273

    def filter_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        if len(frames) == 0:
            return
        if frames[0].ndim != 3:
            raise ValueError("Frame arrays don't have 3 dimensions")

        for idx, frame in enumerate(frames):
            print("Frame:", idx+1)
            width, height, channels = frame.shape
            new_frame = np.ndarray(shape=frame.shape, dtype=frame.dtype)
            kw, kh = self.kernel.shape[0]//2, self.kernel.shape[1]//2

            for x in range(kw, width-kw):
                for y in range(kh, height-kh):
                    pixel = [0, 0, 0]

                    for kern_x in range(-kw, kw):
                        for kern_y in range(-kh, kh):
                            pixel += frame[x+kern_x, y+kern_y]*self.kernel[kh+kern_y][kw+kern_x]

                    new_frame[x, y] = pixel

            yield new_frame
