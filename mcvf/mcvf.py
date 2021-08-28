from typing import List
import cv2
import numpy as np
from mcvf import filters


class Video:
    def __init__(self, fname: str = None):
        self.frames: List[np.ndarray] = []

        if fname is not None:
            self.load_from_file(fname)

    def load_from_file(self, fname: str):
        if len(self.frames) != 0:
            self.frames = []

        cap = cv2.VideoCapture(fname)
        if not cap.isOpened():
            raise FileNotFoundError("Unable to open %s" % fname)

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            self.frames.append(frame)

        cap.release()

    def save_to_file(self, fname: str, fps: int):
        # W, H = 640, 460
        H, W, _ = self.frames[0].shape

        out = cv2.VideoWriter(
            fname, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H)
        )

        for frame in list(self.frames):
            out.write(cv2.resize(frame, (W, H)))
            out.write(frame)

        out.release()

    def play(self):
        for frame in self.frames:
            cv2.imshow("Frame", frame)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        cv2.destroyWindow("Frame")

    def apply_filter(self, filter: filters.Filter):
        self.frames = list(filter.filter_frames(self.frames))
