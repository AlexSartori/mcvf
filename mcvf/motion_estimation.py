'''
Motion estimation utilities
'''

import cv2  # type: ignore # Tell MyPy to ignore missing type hints
import numpy as np
from multiprocessing import Pool

# import numpy.typing as npt
# import nptyping as npt
# from typing import Any, Type, Union


class MotionVector:
    '''
    A Motion Vector describing a detected movement

    Attributes
    ----------
    origin_x : int
            The X coordinate from where the vector originates
    origin_y : int
            The Y coordinate from where the vector originates
    target_x : int
            The X coordinate where the vector ends
    target_y : int
            The Y coordinate where the vector ends
    magnitude : int
            The strength of the detected movement
    '''

    def __init__(self, origin_x: int, origin_y: int, target_x: int, target_y: int, magnitude: int):
        '''
        Parameters
        ---------
        origin_x : int
                The X coordinate from where the vector originates
        origin_y : int
                The Y coordinate from where the vector originates
        target_x : int
                The X coordinate where the vector ends
        target_y : int
                The Y coordinate where the vector ends
        magnitude : int
                The strength of the detected movement
        '''

        self.origin_x: int = origin_x
        self.origin_y: int = origin_y
        self.target_x: int = target_x
        self.target_y: int = target_y
        self.magnitude: int = magnitude

    def __str__(self):
        '''Return a string representation of the vector'''
        return "<%dx%d [%d] %dx%d>" % (self.origin_x, self.origin_y, self.magnitude, self.target_x, self.target_y)

    def __repr__(self):
        '''Return a string representation of the vector'''
        return str(self)


# Type aliases
FrameType = np.ndarray  # npt.NDArray[np.int8]  # npt.NDArray[(Any, Any), np.int8]
MFType = np.ndarray  # npt.NDArray[MotionVector]  # npt.NDArray[(Any, Any), MotionVector]


class BBME:
    '''
    A Block-Based Motion Estimator

    Attributes
    ----------
    block_size : int
        The size in pixels of each block in which frames are subdivided
    window_size : int
        How many neighboring blocks are searched for a match when estimating motion

    '''

    def __init__(self, frames: list[FrameType], block_size: int = 16, window_size: int = 5, algorithm: str = 'EBBME'):
        '''
        Parameters
        ----------
        frames : list[np.ndaray]
            A list of frames to process (as NumPy arrays)
        block_size : int
            The size in pixels of each block in which frames are subdivided
        window_size : int
            How many neighboring blocks are searched for a match when estimating motion
        algorithm : str
            Which algorithm to use to detect motion

        Raises
        ------
        ValueError
            If the requested algorithm is not available
        '''

        self.frames: list[FrameType] = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
        self.block_size: int = block_size
        self.window_size: int = window_size
        if algorithm != 'EBBME':
            raise ValueError("Algorithm not implemented: %s" % algorithm)

    def calculate_motion_field(self):
        '''
        Iterate all frames to estimate a motion field for each one

        Returns
        -------
        motion_field : list[np.ndaray]
            A list of motion fields, one for each frame after the first
        '''

        with Pool() as p:
            return p.map(
                self._calculate_frame_mf,
                [(self.frames[i-1], self.frames[i]) for i in range(1, len(self.frames))]
            )

    def _calculate_frame_mf(self, frames_pair) -> MFType:
        '''
        Subdivide a pair of frames into blocks and compare them to estimate a motion field

        Parameters
        ----------
        frames_pair : tuple[np.ndaray, np.ndaray]
            The two frames from which to extract a motion field

        Returns
        -------
        motion_field : np.ndarray
            A matrix of motion vectors among frame blocks
        '''

        f_ref, f_target = frames_pair
        h, w = f_ref.shape
        bh, bw = h//self.block_size, w//self.block_size
        MF: MFType = np.ndarray(shape=(bh, bw), dtype=MotionVector)

        for bx in range(bw):
            for by in range(bh):
                MF[by, bx] = self._calculate_block_vector(f_ref, f_target, bx, by)

        return MF

    def _calculate_block_vector(self, f_ref: FrameType, f_target: FrameType, block_x: int, block_y: int) -> MotionVector:
        '''
        Calculate the motion vector of a given block between two frames

        Parameters
        ----------
        f_ref : np.ndaray
            The reference frame
        f_target : np.ndaray
            The target frame
        block_x : int
            The X coordinate of the block to analyze
        block_y : int
            The Y coordinate of the block to analyze

        Returns
        -------
        vector : `MotionVector`
            The estimated motion vector for the given block
        '''

        ws = self.window_size
        DFDs: np.ndarray = self._calculate_blocks_DFD(f_ref, f_target, block_x, block_y)
        bs, hbs = self.block_size, self.block_size//2
        min_x, min_y, min_val = ws//2, ws//2, DFDs[ws//2, ws//2]

        for x in range(DFDs.shape[1]):
            for y in range(DFDs.shape[0]):
                if DFDs[y, x] != -1 and DFDs[y, x] < min_val:
                    min_val = DFDs[y, x]
                    min_x = x
                    min_y = y

        return MotionVector(
            block_x*bs+hbs, block_y*bs+hbs,
            (block_x-ws//2+min_x)*bs+hbs, (block_y-ws//2+min_y)*bs+hbs,
            DFDs[min_y, min_x]
        )

    def _calculate_blocks_DFD(self, f_ref: FrameType, f_target: FrameType, block_x: int, block_y: int) -> np.ndarray:
        '''
        Calculate the Displaced Frame Difference (DFD) for each block inside the search window of the given block

        Parameters
        ----------
        f_ref : np.ndaray
            The reference frame
        f_target : np.ndaray
            The target frame
        block_x : int
            The X coordinate of the reference block
        block_y : int
            The Y coordinate of the reference block

        Returns
        -------
        blocks : np.ndaray
            A matrix of DFD values for all blocks in the search window
        '''

        p = 2
        h, w = f_ref.shape
        ws, bs = self.window_size, self.block_size
        bw, bh = w//bs, h//bs
        blocks = np.ndarray(shape=(ws, ws), dtype=int)

        for bx in range(ws):
            for by in range(ws):

                wx = block_x - ws//2 + bx
                wy = block_y - ws//2 + by
                blocks[by, bx] = -1

                if wx < 0 or wx >= bw:
                    continue
                if wy < 0 or wy >= bh:
                    continue

                blocks[by, bx] = 0

                for px_x in range(wx*bs, wx*bs+bs):
                    for px_y in range(wy*bs, wy*bs+bs):
                        blocks[by, bx] += abs(
                            int(f_target[px_y, px_x]) - int(f_ref[px_y, px_x])
                        )**p

        return blocks
