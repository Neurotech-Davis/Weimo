import time
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Loading model
models_dir = Path(__file__).parent.parent / "models"
model_path = str(models_dir / "face_landmarker.task")
# Loading camera correction matrices
cam_matrix_builtin = np.load(models_dir / "intrinsic_built-in.npy")
cam_matrix_new_builtin = np.load(models_dir / "intrinsicNew_built-in.npy")
dist_builtin = np.load(models_dir / "dist_built-in.npy")
cam_matrix_usb = np.load(models_dir / "intrinsic_usb.npy")
cam_matrix_new_usb = np.load(models_dir / "intrinsicNew_usb.npy")
dist_usb = np.load(models_dir / "dist_usb.npy")

FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
options = FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.VIDEO,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1,
)
model_points = np.array(
    [
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye corner
        (225.0, 170.0, -135.0),  # Right eye corner
        (-150.0, -150.0, -125.0),  # Left mouth corner
        (150.0, -150.0, -125.0),  # Right mouth corner
    ],
    dtype="double",
)


SENSITIVITY = 60
PITCH_OFFSET = 0  # Adjust if cursor is too high/low when looking center
YAW_OFFSET = 0
# Define this so that the correct matrix is loaded
BUILTIN_CAMERA_INDEX = 0


def eyetracker_worker(shared_state):
    # SETUP
    shared_state.tracker_running.value = True
    landmarker = FaceLandmarker.create_from_options(options)
    CAMERA_INDEX = shared_state.camera_index.value

    prev_x, prev_y = 0, 0

    def open_camera(index):
        cap = cv2.VideoCapture(index)
        if index != BUILTIN_CAMERA_INDEX:
            K, K_new, dist = cam_matrix_builtin, cam_matrix_new_builtin, dist_builtin
        else:
            K, K_new, dist = cam_matrix_usb, cam_matrix_new_usb, dist_usb
        return cap, K, K_new, dist

    cap, K, K_new, dist = open_camera(CAMERA_INDEX)

    # LOOP
    while not shared_state.shutdown.is_set():
        # hot-swap camera if UI changed the index
        new_index = shared_state.camera_index.value
        if new_index != CAMERA_INDEX:
            cap.release()
            CAMERA_INDEX = new_index
            cap, K, K_new, dist = open_camera(CAMERA_INDEX)

        SMOOTHING = shared_state.smoothing_factor.value

        success, frame = cap.read()
        if not success:
            # print("[eyetracker_worker] Error reading from camera")
            time.sleep(1)
            continue
        
        frame = cv2.undistort(frame, K, dist, None, K_new)
        h, w, _ = frame.shape
        rgb_frame = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        timestamp_ms = int(time.time() * 1000)
        detection_result = landmarker.detect_for_video(rgb_frame, timestamp_ms)

        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]

            # ... your existing processing unchanged ...
            # 1. Extract 2D Image Points from landmarks
            image_points = np.array(
                [
                    (landmarks[1].x * w, landmarks[1].y * h),
                    (landmarks[199].x * w, landmarks[199].y * h),
                    (landmarks[33].x * w, landmarks[33].y * h),
                    (landmarks[263].x * w, landmarks[263].y * h),
                    (landmarks[61].x * w, landmarks[61].y * h),
                    (landmarks[291].x * w, landmarks[291].y * h),
                ],
                dtype="double",
            )

            # 2. Estimate Head Pose (PnP)
            # focal_length = w
            # cam_matrix = np.array(
            #     [[focal_length, 0, w / 2], [0, focal_length, h / 2], [0, 0, 1]],
            #     dtype="double",
            # )

            # success_flag, rot_vec, trans_vec
            _, rot_vec, _ = cv2.solvePnP(model_points, image_points, K_new, None)

            # 3. Get Rotation Angles (Degrees)
            rmat, _ = cv2.Rodrigues(rot_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            pitch, yaw = angles[0], angles[1]

            # Normalize pitch
            # Center is large by default (180/-180 degrees). Need to recenter for sensible math.
            pitch = pitch - 180 if pitch > 0 else pitch + 180

            # 4. Map Rotation to Normalised Coordinates (0.0 - 1.0)
            target_x = 0.5 + (yaw - YAW_OFFSET) / (SENSITIVITY)
            target_y = 0.5 + (pitch - PITCH_OFFSET) / (SENSITIVITY)

            # 5. Exponential Moving Average Smoothing
            curr_x = prev_x + (target_x - prev_x) * SMOOTHING
            curr_y = prev_y + (target_y - prev_y) * SMOOTHING

            # 6. CLAMPING: keep within 0.0 - 1.0
            final_x = float(np.clip(curr_x, 0.0, 1.0))
            final_y = float(np.clip(curr_y, 0.0, 1.0))

            shared_state.set_gaze(final_x, final_y)
            shared_state.face_detected.value = True
            prev_x, prev_y = final_x, final_y
        else:
            shared_state.face_detected.value = False
            shared_state.set_gaze(-1.0, -1.0)

        with shared_state.frame_buffer.get_lock():
            buf = np.frombuffer(shared_state.frame_buffer.get_obj(), dtype=np.uint8)
            buf[:] = frame.flatten()
        shared_state.frame_ready.set()

    # TEARDOWN
    shared_state.tracker_running.value = False
    cap.release()
    landmarker.close()
