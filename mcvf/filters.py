import cv2
from typing import List
import numpy as np
from multiprocessing import Pool

from mcvf import motion_estimation


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
            return []
        if frames[0].ndim != 3:
            raise ValueError("Frame arrays don't have 3 dimensions")
        with Pool() as p:
            return p.map(self._filter_frame, frames)

    def _filter_frame(self, frame):
        new_frame = np.ndarray(shape=frame.shape, dtype=frame.dtype)
        return cv2.GaussianBlur(frame, (5, 5), 1)

        # print("Frame:", idx+1)
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

        return new_frame


class BBMEDrawerFilter(Filter):
    def __init__(self):
        pass

    def filter_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        BBME = motion_estimation.BBME(frames, block_size=10)
        max_mag = 255*BBME.block_size*BBME.block_size/10
        for frame, mf in zip(frames, BBME.calculate_motion_field()):
            new_f = frame

            for row in mf:
                for vector in row:
                    len = vector.magnitude*2/max_mag
                    tx = int(vector.origin_x + (vector.origin_x - vector.target_x)*len)
                    ty = int(vector.origin_y + (vector.origin_y - vector.target_y)*len)

                    new_f = cv2.arrowedLine(
                        new_f,
                        (vector.origin_x, vector.origin_y),
                        (tx, ty),
                        (0, 0, 200),
                        thickness=1,
                        tipLength=0.3
                    )

                yield new_f
