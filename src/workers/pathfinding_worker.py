from pathlib import Path
import time
import sys, os
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from pathfinding.camera_nav import screen_to_location


MOUNT_HEIGHT = 12  # mm (1.2 m)
MOUNT_ANGLE = 0

# Loading camera correction matrices
models_dir = Path(__file__).parent.parent / "models"
cam_matrix_usb = np.load(models_dir / "intrinsic_usb.npy")
cam_matrix_new_usb = np.load(models_dir / "intrinsicNew_usb.npy")
dist_usb = np.load(models_dir / "dist_usb.npy")


def load_calibration():
    # K, K_new, dist
    return cam_matrix_usb, cam_matrix_new_usb, dist_usb


def pathfinding_worker(shared_state):
    # SETUP
    try:
        K, _, dist_coeffs = load_calibration()
        cam_cfg = screen_to_location.CameraConfig(
            height=MOUNT_HEIGHT, angle=MOUNT_ANGLE
        )  # Has to be known beforehand
        img_w, img_h = 640, 480  # Known ahead of time

        shared_state.pathfinding_running.value = True

        # LOOP
        while not shared_state.shutdown.is_set():
            gaze_x, gaze_y = shared_state.get_gaze()

            if gaze_x >= 0 and gaze_y >= 0:
                px = int(gaze_x * img_w)
                py = int(gaze_y * img_h)
                # optical correction first
                distorted = np.array([[[px, py]]], dtype=np.float32)
                undistorted = cv2.undistortPoints(distorted, K, dist_coeffs, P=K)
                px = int(undistorted[0, 0, 0])
                py = int(undistorted[0, 0, 1])

                # then physical trig
                h_angle, dist = screen_to_location.pixel_to_point(px, py, K, cam_cfg)
                shared_state.target_angle.value = float(h_angle)
                shared_state.target_dist.value = float(dist)

            time.sleep(0.05)

    # TEARDOWN
    except Exception as e:
        print(f"[pathfinding_worker] fatal error: {e}")
        shared_state.pathfinding_error.set()

    finally:
        shared_state.pathfinding_running.value = False
        print("[pathfinding_worker] shutdown complete")
