import multiprocessing as mp
import ctypes


class SharedState:
    FRAME_W = 640
    FRAME_H = 480

    def __init__(self):
        # global, defines termination
        self.shutdown = mp.Event()

        ### Eyetracker
        # gaze (eyetracker -> UI)
        self.gaze_x = mp.Value(ctypes.c_float, -1.0)
        self.gaze_y = mp.Value(ctypes.c_float, -1.0)
        self.face_detected = mp.Value(ctypes.c_bool, False)
        self.tracker_running = mp.Value(ctypes.c_bool, False)

        FRAME_W, FRAME_H = 640, 480
        self.frame_buffer = mp.Array(ctypes.c_uint8, FRAME_W * FRAME_H * 3)
        self.frame_ready = mp.Event()

        # eyetracker params (UI -> eyetracker)
        self.camera_index = mp.Value(ctypes.c_int, 64)
        self.smoothing_factor = mp.Value(ctypes.c_float, 0.5)

        ### Classifier
        # classifier (classifier -> UI)
        self.classifier_running = mp.Value(ctypes.c_bool, False)
        self.classification = mp.Value(ctypes.c_uint8, -1)

        # classifier params (UI -> classifier)
        # does it need to pass anything?

    def get_gaze(self) -> tuple:
        return self.gaze_x.value, self.gaze_y.value

    def set_gaze(self, x: float, y: float):
        self.gaze_x.value = x
        self.gaze_y.value = y
