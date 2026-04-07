import multiprocessing as mp
import ctypes


class SharedState:
    def __init__(self):
        # global, defines termination
        self.shutdown = mp.Event()

        # gaze (eyetracker -> UI)
        self.gaze_x = mp.Value(ctypes.c_float, -1.0)
        self.gaze_y = mp.Value(ctypes.c_float, -1.0)
        self.face_detected = mp.Value(ctypes.c_bool, False)
        self.tracker_running = mp.Value(ctypes.c_bool, False)

        # eyetracker params (UI -> eyetracker)
        self.camera_index = mp.Value(ctypes.c_int, 64)
        self.smoothing_factor = mp.Value(ctypes.c_float, 0.5)

    def get_gaze(self) -> tuple:
        return self.gaze_x.value, self.gaze_y.value

    def set_gaze(self, x: float, y: float):
        self.gaze_x.value = x
        self.gaze_y.value = y
