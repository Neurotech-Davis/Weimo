import time
from collections import deque
from typing import final
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURATION ---
# Disable fail-safe to prevent script exit if cursor hits (0,0)
pyautogui.FAILSAFE = False

# Path to your downloaded .task model
model_path = "./mediapipe_model/face_landmarker.task"

BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1,
)

# Screen Dimensions
screen_w, screen_h = pyautogui.size()

# Static 3D model points for Perspective-n-Point (PnP)
# These represent a generic human face in mm
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

# --- CONTROL PARAMETERS ---
SMOOTHING = 0.5
SENSITIVITY = 60  # Multiplier for degrees. Increase for faster cursor.
PITCH_OFFSET = 0  # Adjust if cursor is too high/low when looking center
YAW_OFFSET = 0

prev_x, prev_y = screen_w // 2, screen_h // 2

# --- MAIN EXECUTION ---
with FaceLandmarker.create_from_options(options) as landmarker:
    camera_index = 64  # Using your specific camera index
    cap = cv2.VideoCapture(camera_index)

    running_delta = deque(maxlen=30)
    prev_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Error reading from camera, try switching the camera index above")
            break

        h, w, _ = frame.shape

        # Convert to MediaPipe Image format
        rgb_frame = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )

        timestamp_ms = int(time.time() * 1000)
        detection_result = landmarker.detect_for_video(rgb_frame, timestamp_ms)

        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]

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
            focal_length = w
            cam_matrix = np.array(
                [[focal_length, 0, w / 2], [0, focal_length, h / 2], [0, 0, 1]],
                dtype="double",
            )

            _, rot_vec, trans_vec = cv2.solvePnP(
                model_points, image_points, cam_matrix, np.zeros((4, 1))
            )

            # 3. Get Rotation Angles (Degrees)
            rmat, _ = cv2.Rodrigues(rot_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            pitch, yaw = angles[0], angles[1]

            # Normalize pitch
            # Center is large by default (180/-180 degrees). Need to recenter for sensible math.
            pitch = pitch - 180 if pitch > 0 else pitch + 180

            # 4. Map Rotation to Screen Coordinates
            target_x = screen_w / 2 + (yaw - YAW_OFFSET) * SENSITIVITY
            target_y = screen_h / 2 + (pitch - PITCH_OFFSET) * SENSITIVITY

            # 5. Exponential Moving Average Smoothing
            curr_x = prev_x + (target_x - prev_x) * SMOOTHING
            curr_y = prev_y + (target_y - prev_y) * SMOOTHING

            # 6. CLAMPING: Prevents 'struct.error' by keeping coords within screen limits
            final_x = int(np.clip(curr_x, 0, screen_w - 1))
            final_y = int(np.clip(curr_y, 0, screen_h - 1))

            # Move Mouse
            # print(f"X:{final_x}|Y:{final_y}|Pitch:{pitch}")
            pyautogui.moveTo(final_x, final_y, _pause=False)
            prev_x, prev_y = final_x, final_y

        # FPS Calculation
        curr_time = time.time()
        running_delta.append(curr_time - prev_time)
        prev_time = curr_time
        fps = 1 / (sum(running_delta) / len(running_delta))

        # Visualization
        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Head Tracker Cursor", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()
