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
            raise ValueError("Frame arrays are expected to have 3 dimensions")
        with Pool() as p:
            return p.map(self._filter_frame, frames)

    def _filter_frame(self, frame):
        new_frame = np.ndarray(shape=frame.shape, dtype=frame.dtype)
        return cv2.GaussianBlur(frame, (5, 5), 1)

        # print("Frame:", idx+1)
        height, width, channels = frame.shape
        new_frame = np.ndarray(shape=frame.shape, dtype=frame.dtype)
        kh, kw = self.kernel.shape[0]//2, self.kernel.shape[1]//2

        for x in range(kw, width-kw):
            for y in range(kh, height-kh):
                pixel = [0, 0, 0]

                for kern_x in range(-kw, kw):
                    for kern_y in range(-kh, kh):
                        pixel += frame[y+kern_y, x+kern_x]*self.kernel[kh+kern_y][kw+kern_x]

                new_frame[y, x] = pixel

        return new_frame


'''
    Motion-Compensated Gaussian Filter
'''


class MCGaussianFilter(GaussianFilter):
    def __init__(self):
        super()
        self.block_size = 4
        self.motion_threshold = 0

    def filter_frames(self, frames):
        BBME = motion_estimation.BBME(frames, block_size=self.block_size)
        MF = BBME.calculate_motion_field()
        frames = frames[1:]

        if len(frames) != len(MF):
            print("Size mismatch: %d frames / %d MFs" % (len(frames), len(MF)))

        for frame, mf in zip(frames, MF):
            yield self._filter_frame(frame, mf)

    def _filter_frame(self, frame, mf):
        bh, bw = mf.shape
        bs = self.block_size

        for bx in range(bw):
            for by in range(bh):
                if mf[by, bx].magnitude <= self.motion_threshold:
                    x, y = bx*bs, by*bs
                    # frame[y:y+bs, x:x+bs] = cv2.GaussianBlur(frame[y:y+bs, x:x+bs], (5, 5), 1)
                    # frame[y:y+bs, x:x+bs] = np.zeros(shape=(bs, bs, 3))
                    frame[y:y+bs, x:x+bs] = frame[y:y+bs, x:x+bs]//3

        return frame


class BBMEDrawerFilter(Filter):
    def __init__(self):
        pass

    def filter_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        BBME = motion_estimation.BBME(frames, block_size=15)
        # max_mag = 255*BBME.block_size*BBME.block_size
        max_mag = 1*BBME.block_size*BBME.block_size

        for frame, mf in zip(frames, BBME.calculate_motion_field()):
            new_f = frame

            for row in mf:
                for vector in row:
                    len = 5*vector.magnitude/max_mag
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
