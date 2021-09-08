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
    '''

    def __init__(self, origin_x: int, origin_y: int, target_x: int, target_y: int):
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
        '''

        self.origin_x: int = origin_x
        self.origin_y: int = origin_y
        self.target_x: int = target_x
        self.target_y: int = target_y

    def __str__(self):
        '''Return a string representation of the vector'''
        return "<%dx%d -> %dx%d>" % (self.origin_x, self.origin_y, self.target_x, self.target_y)

    def __repr__(self):
        '''Return a string representation of the vector'''
        return str(self)


# Type aliases
FrameType = np.ndarray  # npt.NDArray[np.int8]  # npt.NDArray[(Any, Any), np.int8]
MFType = np.ndarray  # npt.NDArray[MotionVector]  # npt.NDArray[(Any, Any), MotionVector]

motion_threshold = 12000


class BBME:
    '''
    A Block-Based Motion Estimator

    Attributes
    ----------
    block_size : int
        The size in pixels of each block in which frames are subdivided
    window_size : int
        How many neighboring blocks are searched for a match when estimating motion
    algorithm : str
        The algorithm being used to detect motion
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

        if algorithm not in ['EBBME', '2DLS']:
            raise ValueError("Algorithm not implemented: %s" % algorithm)
        else:
            self.algorithm = algorithm

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
                if self.algorithm == 'EBBME':
                    MF[by, bx] = self._calculate_block_vector_EBBME(f_ref, f_target, bx, by)
                elif self.algorithm == '2DLS':
                    MF[by, bx] = self._calculate_block_vector_2DLS(f_ref, f_target, bx, by)

        return MF

    def _calculate_block_vector_EBBME(self, f_ref: FrameType, f_target: FrameType, bx: int, by: int) -> MotionVector:
        '''
        Calculate the motion vector of a given block between two frames using the Extensive Search algorithm

        Parameters
        ----------
        f_ref : np.ndaray
            The reference frame
        f_target : np.ndaray
            The target frame
        bx : int
            The X coordinate of the block to analyze
        by : int
            The Y coordinate of the block to analyze

        Returns
        -------
        vector : `MotionVector`
            The estimated motion vector for the given block
        '''

        ws = self.window_size
        DFDs: np.ndarray = self._calculate_DFD_matrix_EBBME(f_ref, f_target, bx, by)
        bs, hbs = self.block_size, self.block_size//2
        h, w = f_ref.shape
        bw, bh = w//bs, h//bs
        min_x, min_y, min_val = bx, by, DFDs[by, bx]

        for off_x in range(-ws//2, ws//2 + 1):
            for off_y in range(-ws//2, ws//2 + 1):
                x, y = bx + off_x, by + off_y

                if x < 0 or x >= bw or y < 0 or y >= bh:
                    continue

                if DFDs[y, x] != -1 and DFDs[y, x] < min_val - motion_threshold:
                    min_val = DFDs[y, x]
                    min_x = x
                    min_y = y

        return MotionVector(
            bx*bs+hbs, by*bs+hbs,
            min_x*bs+hbs, min_y*bs+hbs
        )

    def _calculate_block_vector_2DLS(self, f_ref: FrameType, f_target: FrameType, bx: int, by: int) -> MotionVector:
        '''
        Calculate the motion vector of a given block between two frames using the 2D-Log Seach algorithm

        Parameters
        ----------
        f_ref : np.ndaray
            The reference frame
        f_target : np.ndaray
            The target frame
        bx : int
            The X coordinate of the block to analyze
        by : int
            The Y coordinate of the block to analyze

        Returns
        -------
        vector : `MotionVector`
            The estimated motion vector for the given block
        '''

        S = 8
        center_x, center_y = bx, by
        bs, hbs = self.block_size, self.block_size//2
        h, w = f_ref.shape
        bw, bh = w//bs, h//bs

        while True:
            DFDs: np.ndarray = self._calculate_DFD_matrix_2DLS(f_ref, f_target, center_x, center_y)
            min_x, min_y, min_val = center_x, center_y, DFDs[center_y, center_x]

            for off in [(0, -1), (-1, 0), (0, 0), (1, 0), (0, 1)]:
                x = center_x + off[0]
                y = center_y + off[1]

                if x < 0 or x >= bw or y < 0 or y >= bh:
                    continue

                if DFDs[y, x] != -1 and DFDs[y, x] < min_val - motion_threshold:
                    min_val = DFDs[y, x]
                    min_x = x
                    min_y = y

            if S > 1:
                if min_x == center_x and min_y == center_y:
                    S //= 2
                else:
                    center_x, center_y = min_x, min_y
            else:
                break

        return MotionVector(
            bx*bs+hbs, by*bs+hbs,
            min_x*bs+hbs, min_y*bs+hbs
        )

    def _calculate_DFD_matrix_EBBME(self, f_ref: FrameType, f_target: FrameType, bx: int, by: int) -> np.ndarray:
        '''
        Calculate the Displaced Frame Difference (DFD) for each block inside the search window of the given block

        Parameters
        ----------
        f_ref : np.ndaray
            The reference frame
        f_target : np.ndaray
            The target frame
        bx : int
            The X coordinate of the reference block
        by : int
            The Y coordinate of the reference block

        Returns
        -------
        blocks : np.ndaray
            A matrix of DFD values for all blocks in the search window
        '''

        h, w = f_ref.shape
        ws, bs = self.window_size, self.block_size
        bw, bh = w//bs, h//bs
        blocks = np.full(shape=(bh, bw), fill_value=-1, dtype=int)

        for wx in range(bx - ws//2, bx + ws//2 + 1):
            for wy in range(by - ws//2, by + ws//2 + 1):

                if wx < 0 or wx >= bw:
                    continue
                if wy < 0 or wy >= bh:
                    continue

                blocks[wy, wx] = self._calculate_block_DFD(f_ref, f_target, wx, wy)

        return blocks

    def _calculate_DFD_matrix_2DLS(self, f_ref: FrameType, f_target: FrameType, bx: int, by: int) -> np.ndarray:
        '''
        Calculate the Displaced Frame Difference (DFD) for the N,E,W,S directions from the given one

        Parameters
        ----------
        f_ref : np.ndaray
            The reference frame
        f_target : np.ndaray
            The target frame
        bx : int
            The X coordinate of the reference block
        by : int
            The Y coordinate of the reference block

        Returns
        -------
        blocks : np.ndaray
            A matrix of DFD values for the neighbors relevant to the 2DLS algorithm
        '''

        h, w = f_ref.shape
        bs = self.block_size
        bw, bh = w//bs, h//bs
        blocks = np.full(shape=(bh, bw), fill_value=-1, dtype=int)

        for off in [(0, -1), (-1, 0), (0, 0), (1, 0), (0, 1)]:
            x, y = bx + off[0], by + off[1]

            if x < 0 or x >= bw:
                continue
            if y < 0 or y >= bh:
                continue

            blocks[y, x] = self._calculate_block_DFD(f_ref, f_target, x, y)

        return blocks

    def _calculate_block_DFD(self, f_ref: FrameType, f_target: FrameType, bx: int, by: int) -> int:
        '''
        Calculate the Displaced Frame Difference (DFD) of a block between a reference and a target frame

        Parameters
        ----------
        f_ref : np.ndaray
            The reference frame
        f_target : np.ndaray
            The target frame
        bx : int
            The X coordinate of the block
        by : int
            The Y coordinate of the block

        Returns
        -------
        DFD : int
            The Displaced Frame Difference of the indicated block
        '''
        p = 1
        res = 0
        bs = self.block_size
        h, w = f_ref.shape

        for x in range(bx*bs, bx*bs+bs):
            for y in range(by*bs, by*bs+bs):
                if x >= w or y >= h:
                    continue

                res += abs(
                    int(f_target[y, x]) - int(f_ref[y, x])
                )**p

        return res
