'''
Video filters with support for motion-compensation
'''

import cv2
import numpy as np
from multiprocessing import Pool

from mcvf import motion_estimation


class Filter:
    '''
    Base class for video filters
    '''

    def __init__(self):
        pass

    def filter_frames(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        '''
        Parse the given list of frames and return a new list of filtered ones

        Parameters
        ----------
        frames : list[np.ndaray]
            A list of frames (as NumPy arrays) to filter

        Returns
        -------
            frames : list[np.ndarray]
                A list of filtered frames
        '''
        pass


class GaussianFilter(Filter):
    '''
    Low-Pass Gaussian blur
    '''

    def __init__(self):
        self.kernel = np.array([
            [1,  4,  7,  4, 1],
            [4, 16, 26, 16, 4],
            [7, 26, 41, 26, 7],
            [4, 16, 26, 16, 4],
            [1,  4,  7,  4, 1]
        ])/273

    def filter_frames(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        if len(frames) == 0:
            return []
        if frames[0].ndim != 3:
            raise ValueError("Frame arrays are expected to have 3 dimensions")
        with Pool() as p:
            return p.map(self._filter_frame, frames)

    def _filter_frame(self, frame):
        new_frame = np.ndarray(shape=frame.shape, dtype=frame.dtype)

        if True:
            # Faster OpenCV implementation
            return cv2.GaussianBlur(frame, (5, 5), 1)
        else:
            # Slower self-implementation
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


class MCGaussianFilter(GaussianFilter):
    '''
    Motion-Compensated Low-Pass gaussian blur

    Attributes
    ----------
    block_size : int
        The size in pixel of the blocks in which the frames will be subdivided
    motion_threshold : int
        The motion vector strength above which there will be considered to be movement
    '''

    def __init__(self, block_size, motion_threshold):
        '''
        Parameters
        ----------
        block_size : int
            The size in pixel of the blocks in which the frames will be subdivided
        motion_threshold : int
            The motion vector strength above which there will be considered to be movement
        '''

        super()
        self.block_size = block_size
        self.motion_threshold = motion_threshold

    def filter_frames(self, frames):
        BBME = motion_estimation.BBME(frames, block_size=self.block_size)
        MF = BBME.calculate_motion_field()

        if len(frames) - 1 != len(MF):
            raise ValueError("Size mismatch: %d (-1) frames / %d MFs" % (len(frames), len(MF)))

        yield frames[0]
        for frame, mf in zip(frames[1:], MF):
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


class MFDrawerFilter(Filter):
    '''
    A drawer filter to render motion vectors onto each frame

    Attributes
    ----------
    block_size : int
        The size in pixel of the blocks in which the frames will be subdivided
    '''

    def __init__(self, block_size):
        '''
        Parameters
        ----------
        block_size : int
            The size in pixel of the blocks in which the frames will be subdivided
        '''
        self.block_size = block_size

    def filter_frames(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        BBME = motion_estimation.BBME(frames, block_size=self.block_size)
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
