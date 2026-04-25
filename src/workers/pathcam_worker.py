import sys
import time
import cv2
import numpy as np
from pathlib import Path


# Loading correction matrices
models_dir = Path(__file__).parent.parent / "models"
cam_matrix_builtin = np.load(models_dir / "intrinsic_built-in.npy")
cam_matrix_new_builtin = np.load(models_dir / "intrinsicNew_built-in.npy")
dist_builtin = np.load(models_dir / "dist_built-in.npy")
cam_matrix_usb = np.load(models_dir / "intrinsic_usb.npy")
cam_matrix_new_usb = np.load(models_dir / "intrinsicNew_usb.npy")
dist_usb = np.load(models_dir / "dist_usb.npy")

BUILTIN_CAMERA_INDEX = 0


def get_correct_matrices(index):
    # returns K, K_new, dist
    if index == BUILTIN_CAMERA_INDEX:
        return cam_matrix_builtin, cam_matrix_new_builtin, dist_builtin
    return cam_matrix_usb, cam_matrix_new_usb, dist_usb


def pathcam_worker(shared_state):
    # SETUP
    shared_state.pathcam_running.value = True
    CAMERA_INDEX = shared_state.pathcam_index.value
    CAM_BACKEND = cv2.CAP_V4L2 if shared_state.on_linux else cv2.CAP_DSHOW

    def open_camera(index):
        cap = cv2.VideoCapture(index, CAM_BACKEND)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        K, K_new, dist = get_correct_matrices(index)
        W, H = shared_state.PATH_FRAME_W, shared_state.PATH_FRAME_H
        map1, map2 = cv2.initUndistortRectifyMap(
            K, dist, None, K_new, (W, H), cv2.CV_32FC1
        )
        return cap, map1, map2

    cap, map1, map2 = open_camera(CAMERA_INDEX)

    # LOOP
    try:
        while not shared_state.shutdown.is_set():
            # hot-swap camera if UI changed the index
            new_index = shared_state.pathcam_index.value
            if new_index != CAMERA_INDEX:
                cap.release()
                CAMERA_INDEX = new_index
                cap, map1, map2 = open_camera(CAMERA_INDEX)

            success, frame = cap.read()
            if not success:
                # print("[pathcam_worker] Error reading from camera")
                time.sleep(1)
                continue

            undistorted_frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

            with shared_state.path_frame_buffer.get_lock():
                buf = np.frombuffer(
                    shared_state.path_frame_buffer.get_obj(), dtype=np.uint8
                )
                buf[:] = undistorted_frame.flatten()
            shared_state.path_frame_ready.set()

    except Exception as e:
        print(f"[pathcam_worker] Fatal error: {e}")
    finally:
        # TEARDOWN
        shared_state.pathcam_running.value = False
        cap.release()
