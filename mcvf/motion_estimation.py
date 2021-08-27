import cv2
from typing import List
import numpy as np
from multiprocessing import Pool


class MotionVector:
    def __init__(self, origin_x: int, origin_y: int, target_x: int, target_y: int, magnitude: int):
        self.origin_x: int = origin_x
        self.origin_y: int = origin_y
        self.target_x: int = target_x
        self.target_y: int = target_y
        self.magnitude: int = magnitude

    def __str__(self):
        return "<%dx%d [%d] %dx%d>" % (self.origin_x, self.origin_y, self.magnitude, self.target_x, self.target_y)

    def __repr__(self):
        return str(self)


class BBME:
    def __init__(self, frames: List[np.ndarray], block_size: int = 16, algorithm: str = 'EBBME'):
        self.frames: List[np.ndarray] = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
        self.block_size: int = block_size
        if algorithm != 'EBBME':
            raise ValueError("Algorithm not implemented: %s" % algorithm)

    def calculate_motion_field(self):
        res = []

        with Pool() as p:
            res = [] + p.map(
                self._calculate_frame_mf,
                [(self.frames[i-1], self.frames[i]) for i in range(1, len(self.frames))]
            )

        return res

        # f_prev = None
        # for i, f in enumerate(self.frames):
        #     print("%.2f%% (%d/%d)" % (100*i/len(self.frames), i, len(self.frames)), end='\r')
        #
        #     if f_prev is None:
        #         yield []
        #     else:
        #         yield self._calculate_frame_mf((f_prev, f))
        #     f_prev = f
        # print()

    # def _calculate_frame_mf(self, f_ref: np.ndarray, f_target: np.ndarray) -> List[MotionVector]:
    def _calculate_frame_mf(self, args) -> np.ndarray:
        f_ref, f_target = args
        h, w = f_ref.shape
        bw, bh = w//self.block_size, h//self.block_size
        MF: np.ndarray = np.ndarray(shape=(bw, bh), dtype=MotionVector)

        # with Pool() as p:
        #     MF = p.map(self._calculate_block_vector, [(f_ref, f_target, x, y) for x in range(bw) for y in range(bh)])

        for bx in range(bw):
            for by in range(bh):
                MF[bx, by] = self._calculate_block_vector(f_ref, f_target, bx, by)

        return MF

    def _calculate_block_vector(self, f_ref: np.ndarray, f_target: np.ndarray, block_x: int, block_y: int) -> MotionVector:
        ws = 5
        DFDs: np.ndarray = self._calculate_blocks_DFD(f_ref, f_target, block_x, block_y, ws)
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

    def _calculate_blocks_DFD(self, f_ref: np.ndarray, f_target: np.ndarray, block_x: int, block_y: int, ws: int) -> np.ndarray:
        h, w = f_ref.shape
        bs, bw, bh = self.block_size, w//self.block_size, h//self.block_size
        blocks = np.ndarray(shape=(ws, ws), dtype=int)

        for bx in range(ws):
            for by in range(ws):

                wx = block_x - ws + bx
                wy = block_y - ws + by
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
                        )

        return blocks
