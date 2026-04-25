import sys
import multiprocessing as mp
import ctypes


class SharedState:
    FRAME_W = 640
    FRAME_H = 480
    usb_is_main_cam = "--usb-main" in sys.argv
    on_linux = "--linux" in sys.argv

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

        self.EYE_FRAME_W = 640
        self.EYE_FRAME_H = 480
        self.eye_frame_buffer = mp.Array(
            ctypes.c_uint8, self.EYE_FRAME_W * self.EYE_FRAME_H * 3
        )
        self.eye_frame_ready = mp.Event()

        # eyetracker params (UI -> eyetracker)
        if self.usb_is_main_cam:
            self.eye_camera_index = mp.Value(ctypes.c_int, 64)
        else:
            self.eye_camera_index = mp.Value(ctypes.c_int, 1)
        self.smoothing_factor = mp.Value(ctypes.c_float, 0.5)

        ### ---------------------
        ### Pathfinding cam worker
        ### ---------------------
        self.pathcam_running = mp.Value(ctypes.c_bool, False)
        self.PATH_FRAME_W = 640
        self.PATH_FRAME_H = 480
        self.path_frame_buffer = mp.Array(
            ctypes.c_uint8, self.PATH_FRAME_W * self.PATH_FRAME_H * 3
        )
        self.path_frame_ready = mp.Event()
        self.pathcam_index = mp.Value(ctypes.c_int, 0)

        ### -----------
        ### Classifier
        ### -----------
        # classifier (classifier -> UI)
        self.classifier_running = mp.Value(ctypes.c_bool, False)
        self.prediction = mp.Value(ctypes.c_int, -1)
        self.pred_confidence = mp.Value(ctypes.c_float, -1.0)
        self.mock_classifier = mp.Value(ctypes.c_bool, False)   # set by main_process
        self.demo_classifier = mp.Value(ctypes.c_bool, False)   # set by main_process
        # When True, classifier_worker skips writing predictions (demo keyboard takes over)
        self.demo_override = mp.Value(ctypes.c_bool, False)

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
        # Motor <-> UI
        # shared_state.py
        self.turn_command = mp.Value(ctypes.c_float, 0.0)  # 0 = no turn, else degrees

        ### -----------
        ### Pathfinding
        ### -----------
        # UI -> Pathfinding
        self.pathfinding_camera_index = mp.Value(ctypes.c_int, 0)
        # Pathfinding -> UI
        self.pathfinding_running = mp.Value(ctypes.c_bool, False)
        self.pathfinding_error = mp.Event()

        # Pathfinding -> Motor (lidar guard)
        self.obstacle_detected = mp.Value(ctypes.c_bool, False)
        self.lidar_dist = mp.Value(ctypes.c_float, float("inf"))

        ### -----------
        ### LIDAR
        ### -----------
        self.lidar_running = mp.Value(ctypes.c_bool, False)

        ### -----------
        ### VLM Safety
        ### -----------
        self.vlm_running = mp.Value(ctypes.c_bool, False)
        self.vlm_last_verdict = mp.Array(ctypes.c_char, 255) # Store response string
        self.vlm_is_busy = mp.Value(ctypes.c_bool, False)
        # To avoid double-triggering the same move
        self.vlm_processed_dist = mp.Value(ctypes.c_float, 0.0)
        
    def get_gaze(self) -> tuple:
        return self.gaze_x.value, self.gaze_y.value

    def set_gaze(self, x: float, y: float):
        self.gaze_x.value = x
        self.gaze_y.value = y