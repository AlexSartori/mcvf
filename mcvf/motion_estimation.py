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
        f_prev = None
        for f in self.frames:
            if f_prev is not None:
                yield self._calculate_frame_mf(f_prev, f)
            f_prev = f

    def _calculate_frame_mf(self, f_ref: np.ndarray, f_target: np.ndarray) -> List[MotionVector]:
        w, h = f_ref.shape
        bw, bh = w//self.block_size, h//self.block_size
        MF: List[MotionVector] = []

        # with Pool() as p:
        #     MF = p.map(self._calculate_block_vector, [(f_ref, f_target, x, y) for x in range(bw) for y in range(bh)])

        for bx in range(bw):
            for by in range(bh):
                MF.append(
                    self._calculate_block_vector(f_ref, f_target, bx, by)
                )

        return MF

    def _calculate_block_vector(self, f_ref: np.ndarray, f_target: np.ndarray, block_x: int, block_y: int) -> MotionVector:
        ws = 3
        DFDs: np.ndarray = self._calculate_blocks_DFD(f_ref, f_target, block_x, block_y, ws)
        bs, hbs = self.block_size, self.block_size//2
        min_x, min_y, min_val = ws//2, ws//2, DFDs[ws//2, ws//2]

        for x in range(DFDs.shape[0]):
            for y in range(DFDs.shape[1]):
                if DFDs[x, y] != -1 and DFDs[x, y] < min_val:
                    min_val = DFDs[x, y]
                    min_x = x
                    min_y = y

        return MotionVector(
            block_x*bs+hbs, block_y*bs+hbs,
            (block_x-ws//2+min_x)*bs+hbs, (block_y-ws//2+min_y)*bs+hbs,
            DFDs[min_x, min_y]
        )

    def _calculate_blocks_DFD(self, f_ref: np.ndarray, f_target: np.ndarray, block_x: int, block_y: int, ws: int) -> np.ndarray:
        w, h = f_ref.shape
        bs, bw, bh = self.block_size, w//self.block_size, h//self.block_size
        blocks = np.ndarray(shape=(ws, ws), dtype=int)

        for bx in range(ws):
            for by in range(ws):

                wx = block_x - ws + bx
                wy = block_y - ws + by
                blocks[bx, by] = -1

                if wy < 0 or wy >= bh:
                    continue
                if wx < 0 or wx >= bw:
                    continue

                blocks[bx, by] = 0

                for px_x in range(wx*bs, wx*bs+bs):
                    for px_y in range(wy*bs, wy*bs+bs):
                        blocks[bx, by] += abs(
                            int(f_ref[px_x, px_y]) - int(f_target[px_x, px_y])
                        )

        return blocks
