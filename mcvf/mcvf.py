import cv2
from mcvf import Filters


class Video:
    def __init__(self, fname: str = None):
        self.cap = None
        if fname is not None:
            self.load_from_file(fname)

    def load_from_file(self, fname: str):
        cap: cv2.VideoCapture = cv2.VideoCapture(fname)

        if not cap.isOpened():
            raise FileNotFoundError("Unable to open %s" % fname)
        else:
            self.cap = cap

    def play(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                cv2.imshow("Frame", frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    def apply_filter(self, filter: Filters.Filter):
        pass
