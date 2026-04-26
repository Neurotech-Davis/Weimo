# calibrate.py
import time
import json
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ── paths (same as your worker) ──────────────────────────────────────────────
models_dir = Path(__file__).parent.parent / "models"
model_path = str(models_dir / "face_landmarker.task")
cam_matrix = np.load(models_dir / "intrinsic_built-in.npy")
cam_matrix_new = np.load(models_dir / "intrinsicNew_built-in.npy")
dist = np.load(models_dir / "dist_built-in.npy")

CALIBRATIONS_DIR = Path(__file__).parent.parent / "calibrations"
CALIBRATIONS_DIR.mkdir(exist_ok=True)

# ── same constants as worker ──────────────────────────────────────────────────
ANCHOR_INDICES = [1, 199, 33, 263, 61, 291]
model_points = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ],
    dtype="double",
)

N_CALIBRATION_FRAMES = 90  # ~3 seconds at 30fps
CAMERA_INDEX = 0
W, H = 640, 480
# ─────────────────────────────────────────────────────────────────────────────


def collect_center_offset(user_id: str):
    # ── landmarker setup ──────────────────────────────────────────────────────
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    # ── camera setup ─────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)  # swap to CAP_V4L2 on linux
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

    map1, map2 = cv2.initUndistortRectifyMap(
        cam_matrix, dist, None, cam_matrix_new, (W, H), cv2.CV_32FC1
    )

    pitches, yaws = [], []
    print(f"\nLook straight at the screen center.")
    print(f"Collecting {N_CALIBRATION_FRAMES} frames — hold still...\n")

    while len(pitches) < N_CALIBRATION_FRAMES:
        ret, frame = cap.read()
        if not ret:
            continue

        undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        rgb = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB),
        )
        result = landmarker.detect_for_video(rgb, int(time.time() * 1000))

        if not result.face_landmarks:
            cv2.putText(
                undistorted,
                "No face detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            cv2.imshow("Calibration", undistorted)
            cv2.waitKey(1)
            continue

        lms = result.face_landmarks[0]
        image_points = np.array(
            [(lms[i].x * W, lms[i].y * H) for i in ANCHOR_INDICES],
            dtype="double",
        )

        _, rot_vec, _ = cv2.solvePnP(model_points, image_points, cam_matrix_new, None)
        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, *_ = cv2.RQDecomp3x3(rmat)
        pitch, yaw = angles[0], angles[1]
        pitch = pitch - 180 if pitch > 0 else pitch + 180

        pitches.append(pitch)
        yaws.append(yaw)

        # live progress
        n = len(pitches)
        bar = "#" * (n * 20 // N_CALIBRATION_FRAMES)
        cv2.putText(
            undistorted,
            f"Calibrating... {n}/{N_CALIBRATION_FRAMES}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Calibration", undistorted)
        cv2.waitKey(1)
        print(f"\r[{bar:<20}] {n}/{N_CALIBRATION_FRAMES}", end="", flush=True)

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()

    pitch_offset = float(np.mean(pitches))
    yaw_offset = float(np.mean(yaws))
    std_pitch = float(np.std(pitches))
    std_yaw = float(np.std(yaws))

    print(f"\n\nCalibration complete.")
    print(f"  pitch_offset: {pitch_offset:.2f}°  (std: {std_pitch:.2f}°)")
    print(f"  yaw_offset:   {yaw_offset:.2f}°  (std: {std_yaw:.2f}°)")

    # high std = user was moving too much, warn them
    if std_pitch > 3.0 or std_yaw > 3.0:
        print("  ⚠️  High variance detected — try again and hold still.")

    out = {
        "user_id": user_id,
        "pitch_offset": pitch_offset,
        "yaw_offset": yaw_offset,
        "std_pitch": std_pitch,
        "std_yaw": std_yaw,
    }
    out_path = CALIBRATIONS_DIR / f"{user_id}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"  Saved → {out_path}")
    return out


if __name__ == "__main__":
    import sys

    uid = sys.argv[1] if len(sys.argv) > 1 else "default"
    collect_center_offset(uid)
