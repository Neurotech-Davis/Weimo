import sys
import multiprocessing as mp
import ctypes


class SharedState:
    FRAME_W = 640
    FRAME_H = 480
    usb_is_main_cam = "--usb-main" in sys.argv

    def __init__(self):
        # global, defines termination
        self.shutdown = mp.Event()

        ### -----------
        ### Eyetracker
        ### -----------
        # gaze (eyetracker -> UI)
        self.gaze_x = mp.Value(ctypes.c_float, -1.0)
        self.gaze_y = mp.Value(ctypes.c_float, -1.0)
        self.face_detected = mp.Value(ctypes.c_bool, False)
        self.tracker_running = mp.Value(ctypes.c_bool, False)

        FRAME_W, FRAME_H = 640, 480
        self.frame_buffer = mp.Array(ctypes.c_uint8, FRAME_W * FRAME_H * 3)
        self.frame_ready = mp.Event()

        # eyetracker params (UI -> eyetracker)
        if self.usb_is_main_cam:
            self.camera_index = mp.Value(ctypes.c_int, 64)
        else:
            self.camera_index = mp.Value(ctypes.c_int, 0)
        self.smoothing_factor = mp.Value(ctypes.c_float, 0.5)

        ### -----------
        ### Classifier
        ### -----------
        # classifier (classifier -> UI)
        self.classifier_running = mp.Value(ctypes.c_bool, False)
        self.prediction = mp.Value(ctypes.c_int, -1)
        self.pred_confidence = mp.Value(ctypes.c_float, -1.0)
        self.mock_classifier = mp.Value(ctypes.c_bool, False)  # set by main_process

        # classifier params (UI -> classifier)
        # does it need to pass anything?

        ### -----------
        ### Motor
        ### -----------
        # Pathfinding -> Motor
        self.motor_command = mp.Value(ctypes.c_int, 0)  # UI writes intent
        self.target_angle = mp.Value(ctypes.c_float, 0.0)
        self.target_dist = mp.Value(ctypes.c_float, 0.0)
        # Motor -> UI
        self.motor_state = mp.Value(ctypes.c_int, 0)
        self.motor_running = mp.Value(ctypes.c_bool, False)
        self.motor_error = mp.Event()

        ### -----------
        ### Pathfinding
        ### -----------
        # UI -> Pathfinding
        self.pathfinding_camera_index = mp.Value(ctypes.c_int, 1)
        # Pathfinding -> UI
        self.pathfinding_running = mp.Value(ctypes.c_bool, False)
        self.pathfinding_error = mp.Event()

        # Pathfinding -> Motor (lidar guard)
        self.obstacle_detected = mp.Value(ctypes.c_bool, False)
        self.lidar_dist = mp.Value(ctypes.c_float, float("inf"))

    def get_gaze(self) -> tuple:
        return self.gaze_x.value, self.gaze_y.value

    def set_gaze(self, x: float, y: float):
        self.gaze_x.value = x
        self.gaze_y.value = y
