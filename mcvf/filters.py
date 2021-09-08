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
        Parse the given list of frames contextually with a Motion Field and
        return a new list of filtered ones

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
        '''
        Apply a gaussian blur with  5x5 kernel

        Parameters
        ----------
        frames : list[np.ndaray]
            A list of frames (as NumPy arrays) to filter

        Returns
        -------
            frames : list[np.ndarray]
                A list of filtered frames
        '''

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
        '''
        Apply a 5x5 gaussian blur to the frames where motion is not present

        Parameters
        ----------
        frames : list[np.ndaray]
            A list of frames (as NumPy arrays) to filter

        Returns
        -------
            frames : list[np.ndarray]
                A list of filtered frames
        '''

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
                if v.origin_x == v.target_x and v.origin_y == v.target_y:
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
        '''
        Darken the frame areas where motion is not present

        Parameters
        ----------
        frames : list[np.ndaray]
            A list of frames (as NumPy arrays) to filter

        Returns
        -------
            frames : list[np.ndarray]
                A list of filtered frames
        '''

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
        '''
        Overlay a needle diagram to each frame showing its motion field

        Parameters
        ----------
        frames : list[np.ndaray]
            A list of frames (as NumPy arrays) to filter

        Returns
        -------
            frames : list[np.ndarray]
                A list of filtered frames
        '''

        BBME = motion_estimation.BBME(
            frames,
            block_size=self.block_size,
            window_size=15,
            algorithm='2DLS'
        )

        yield frames[0]

        for frame, mf in zip(frames[1:], BBME.calculate_motion_field()):
            new_f = frame

            for row in mf:
                for vector in row:
                    new_f = cv2.arrowedLine(
                        new_f,
                        (vector.origin_x, vector.origin_y),
                        (vector.target_x, vector.target_y),
                        (0, 0, 200),
                        thickness=1,
                        tipLength=0.2
                    )

            yield new_f


class MCMovingAvergeFilter(MCFilter):
    def __init__(self, block_size: int):
        super().__init__(block_size)

    def filter_frames(self, frames: list[np.ndarray]) -> Iterable[np.ndarray]:
        BBME = motion_estimation.BBME(
            frames,
            block_size=self.block_size,
            window_size=15,
            algorithm='2DLS'
        )

        self.frames: list[np.ndarray] = list(frames)
        MF = BBME.calculate_motion_field()
        self.mf_map: list[dict] = list([{}] + [self._map_MF(mf) for mf in MF])

        with Pool() as p:
            return [self.frames[0]] + p.map(
                self._filter_frame,
                [i for i in range(1, len(self.frames))]
            )

        # for i, f in enumerate(frames):
        #     print("%.2f%%" % (100*i/len(self.frames)), end='\r')
        #     if i == 0:
        #         continue
        #     yield self._filter_frame(i)
        # print()

    def _map_MF(self, mf: list[np.ndarray]) -> dict:
        mf_map = {}
        bs = self.block_size

        for row in mf:
            for v in row:
                A = (v.origin_x//bs, v.origin_y//bs)
                B = (v.target_x//bs, v.target_y//bs)

                if A[0] != B[0] or A[1] != B[1]:
                    mf_map[B] = A

        return mf_map

    def _filter_frame(self, f_idx: int) -> np.ndarray:
        h, w, c = self.frames[0].shape
        new_f = np.ndarray(shape=(h, w, c), dtype=self.frames[0].dtype)

        for y in range(h):
            for x in range(w):
                new_f[y, x] = self._filter_pixel(f_idx, x, y)

        return new_f

    def _filter_pixel(self, f_idx: int, x: int, y: int) -> np.ndarray:
        alpha = 0.2
        N = min(4, f_idx)
        res = self.frames[f_idx][y, x] * (1 - alpha)

        tmp = np.array([0, 0, 0])
        target = (x, y)

        for n in range(N):
            if target in self.mf_map[f_idx - n]:
                target = self.mf_map[f_idx - n][target]
            tx, ty = target

            for ch in range(3):
                tmp[ch] += self.frames[f_idx - n - 1][ty, tx][ch] * 1/N

        res += tmp * alpha

        return res
