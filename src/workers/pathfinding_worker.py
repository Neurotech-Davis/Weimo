from multiprocessing import get_all_start_methods
from pathlib import Path
import time
import sys
import os
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from pathfinding.camera_nav import screen_to_location


# Loading camera correction matrices
models_dir = Path(__file__).parent.parent / "models"
cam_matrix_usb = np.load(models_dir / "intrinsic_usb.npy")
cam_matrix_new_usb = np.load(models_dir / "intrinsicNew_usb.npy")
dist_usb = np.load(models_dir / "dist_usb.npy")


X_GUTTER = 0.20  # 15% on each side for UI + 5% margin
Y_GUTTER = 0.30  # 25% for top/bottom button height + 5% margin


def is_gaze_in_ui(gx, gy):
    x_in_gutter = gx <= X_GUTTER or gx >= (1.0 - X_GUTTER)
    y_in_gutter = gy <= Y_GUTTER or gy >= (1.0 - Y_GUTTER)
    return x_in_gutter and y_in_gutter


def load_calibration():
    # K, K_new, dist
    return cam_matrix_usb, cam_matrix_new_usb, dist_usb


def pathfinding_worker(shared_state):
    # SETUP

    ## calibrate in /pathfinding/camera_nav/nav_debug_gui.py
    MOUNT_HEIGHT = shared_state.mount_height
    MOUNT_ANGLE = shared_state.mount_angle

    try:
        _, K_new, _ = load_calibration()
        cam_cfg = screen_to_location.CameraConfig(
            height=MOUNT_HEIGHT, angle=MOUNT_ANGLE
        )  # Has to be known beforehand
        img_w = shared_state.PATH_FRAME_W
        img_h = shared_state.PATH_FRAME_H

        shared_state.pathfinding_running.value = True

        # LOOP
        while not shared_state.shutdown.is_set():
            gaze_x, gaze_y = shared_state.get_gaze()
            gaze_in_ui = is_gaze_in_ui(gaze_x, gaze_y)

            if gaze_x >= 0 and gaze_y >= 0 and not gaze_in_ui:
                px = int(gaze_x * img_w)
                py = int(gaze_y * img_h)
                # then physical trig
                h_angle, dist = screen_to_location.pixel_to_point(
                    px, py, K_new, cam_cfg
                )
                shared_state.target_angle.value = float(h_angle)
                shared_state.target_dist.value = float(dist)
            else:
                # no face detected — clear target so motor doesn't act on stale values
                shared_state.target_angle.value = 0.0
                shared_state.target_dist.value = 0.0
            time.sleep(0.05)

    # TEARDOWN
    except Exception as e:
        print(f"[pathfinding_worker] fatal error: {e}")
        shared_state.pathfinding_error.set()

    finally:
        shared_state.pathfinding_running.value = False
        print("[pathfinding_worker] shutdown complete")
