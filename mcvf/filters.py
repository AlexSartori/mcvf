'''
Video filters with support for motion-compensation
'''

import cv2  # type: ignore # Tell MyPy to ignore missing type hints
import numpy as np
from multiprocessing import Pool
from typing import Iterable

from mcvf import motion_estimation


class Filter:
    '''
    Base class for video filters
    '''

    def __init__(self):
        pass

    def filter_frames(self, frames: list[np.ndarray]) -> Iterable[np.ndarray]:
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


class MCFilter(Filter):
    '''
    Base class for motion-compensated video filters
    '''

    def __init__(self, block_size: int):
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

    def filter_frames(self, frames: list[np.ndarray]) -> Iterable[np.ndarray]:
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

    def filter_frames(self, frames: list[np.ndarray]) -> Iterable[np.ndarray]:
        if len(frames) == 0:
            return []
        if frames[0].ndim != 3:
            raise ValueError("Frame arrays are expected to have 3 dimensions")
        with Pool() as p:
            return p.map(self._filter_frame, frames)

    def _filter_frame(self, frame: np.ndarray):
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


class MCGaussianFilter(MCFilter):
    '''
    Motion-Compensated Low-Pass gaussian blur
    '''

    def __init__(self, block_size: int):
        super().__init__(block_size)

    def filter_frames(self, frames: list[np.ndarray]) -> Iterable[np.ndarray]:
        BBME = motion_estimation.BBME(frames, block_size=self.block_size)
        MF = BBME.calculate_motion_field()

        if len(frames) - 1 != len(MF):
            raise ValueError("Size mismatch: %d (-1) frames / %d MFs" % (len(frames), len(MF)))

        yield frames[0]
        for frame, mf in zip(frames[1:], MF):
            yield self._filter_frame(frame, mf)

    def _filter_frame(self, frame: np.ndarray, mf: np.ndarray):
        bh, bw = mf.shape
        bs = self.block_size

        for bx in range(bw):
            for by in range(bh):
                v = mf[by, bx]
                if v.origin_x != v.target_x and v.origin_y != v.target_y:
                    x, y = bx*bs, by*bs
                    frame[y:y+bs, x:x+bs] = cv2.GaussianBlur(frame[y:y+bs, x:x+bs], (5, 5), 1)

        return frame


class MCDarkenFilter(MCFilter):
    '''
    Motion-Compensated darkening filter
    '''

    def __init__(self, block_size: int):
        super().__init__(block_size)

    def filter_frames(self, frames: list[np.ndarray]) -> Iterable[np.ndarray]:
        BBME = motion_estimation.BBME(
            frames,
            block_size=self.block_size,
            window_size=3,
            algorithm='EBBME'
        )
        MF = BBME.calculate_motion_field()

        if len(frames) - 1 != len(MF):
            raise ValueError("Size mismatch: %d (-1) frames / %d MFs" % (len(frames), len(MF)))

        yield frames[0]
        for frame, mf in zip(frames[1:], MF):
            yield self._filter_frame(frame, mf)

    def _filter_frame(self, frame: np.ndarray, mf: np.ndarray):
        bh, bw = mf.shape
        bs = self.block_size

        for bx in range(bw):
            for by in range(bh):
                v = mf[by, bx]
                if v.origin_x == v.target_x and v.origin_y == v.target_y:
                    x, y = bx*bs, by*bs
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

    def __init__(self, block_size: int):
        '''
        Parameters
        ----------
        block_size : int
            The size in pixel of the blocks in which the frames will be subdivided
        '''
        self.block_size = block_size

    def filter_frames(self, frames: list[np.ndarray]) -> Iterable[np.ndarray]:
        BBME = motion_estimation.BBME(
            frames,
            block_size=self.block_size,
            window_size=3,
            algorithm='2DLS'
        )

        for frame, mf in zip(frames, BBME.calculate_motion_field()):
            new_f = frame

            for row in mf:
                for vector in row:
                    print(vector)
                    new_f = cv2.arrowedLine(
                        new_f,
                        (vector.origin_x, vector.origin_y),
                        (vector.target_x, vector.target_y),
                        (0, 0, 200),
                        thickness=1,
                        tipLength=0.1
                    )

                yield new_f
