import time
from collections import deque

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Configuration
model_path = "./mediapipe_model/face_landmarker.task"  # Path to downloaded model
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

with FaceLandmarker.create_from_options(options) as landmarker:
    # ------------------ #
    # This should usually be 0, but my computer has camera on index 64
    camera_index = 64
    #
    # ------------------ #

    cap = cv2.VideoCapture(camera_index)

    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera Resolution: {width}x{height}")

        # Get and print the configured/reported FPS
        fps_configured = cap.get(cv2.CAP_PROP_FPS)
        print(f"Configured FPS (via cap.get()): {fps_configured}")

    # performance tracking
    frame_times = deque(maxlen=30)
    prev_time = 0

    while cap.isOpened():
        # print("hi")
        success, frame = cap.read()
        if not success:
            print("Error reading from camera, try switching the camera index above")
            break

        # MediaPipe expects RGB
        rgb_frame = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )

        # Process the frame
        timestamp_ms = int(time.time() * 1000)
        detection_result = landmarker.detect_for_video(rgb_frame, timestamp_ms)

        if detection_result.face_landmarks:
            # The Face Landmarker result contains 478 landmarks.
            # Landmarks 468-472: Left Iris center and contour
            # Landmarks 473-477: Right Iris center and contour
            landmarks = detection_result.face_landmarks[0]

            for idx in range(468, 478):  # Visualize Iris landmarks
                pt = landmarks[idx]
                x = int(pt.x * frame.shape[1])
                y = int(pt.y * frame.shape[0])
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        curr_time = time.time()
        frame_times.append(curr_time - prev_time)
        prev_time = curr_time
        avg_delta = sum(frame_times) / len(frame_times)
        fps = 1 / avg_delta

        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (20, 70),  # coords
            cv2.FONT_HERSHEY_SIMPLEX,  # font
            1,  # font size
            (0, 255, 0),  # color (green)
            2,  # thickness
        )

        cv2.imshow("Iris Accuracy Demo", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
