# calibrate.py
import time
import json
import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtGui import QPainter, QColor, QFont
from PyQt6.QtCore import Qt

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

on_linux = "--linux" in sys.argv
N_CALIBRATION_FRAMES = 90  # ~3 seconds at 30fps
CAMERA_INDEX = 64 if on_linux else 0
W, H = 640, 480
# ─────────────────────────────────────────────────────────────────────────────


def get_correct_matrices(index):
    # returns K, K_new, dist
    if index == 1:
        return cam_matrix_builtin, cam_matrix_new_builtin, dist_builtin
    return cam_matrix_usb, cam_matrix_new_usb, dist_usb


def collect_center_offset(user_id: str, app):
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
    BACKEND = cv2.CAP_V4L2 if on_linux else cv2.CAP_DSHOW
    cap = cv2.VideoCapture(CAMERA_INDEX, BACKEND)  # swap to CAP_V4L2 on linux

    cam_matrix, cam_matrix_new, dist = get_correct_matrices(CAMERA_INDEX)

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
        app.processEvents()

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


def show_calibration_target():
    app = QApplication.instance() or QApplication(sys.argv)

    class CalibrationTarget(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowFlags(
                Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
            )
            self.showFullScreen()

        def paintEvent(self, event):
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor("black"))

            cx, cy = self.width() // 2, self.height() // 2
            r = 20
            painter.setBrush(QColor("red"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(cx - r, cy - r, r * 2, r * 2)

            painter.setPen(QColor("white"))
            painter.setFont(QFont("Arial", 24))
            painter.drawText(
                0,
                cy + 60,
                self.width(),
                40,
                Qt.AlignmentFlag.AlignHCenter,
                "Look at the dot, hold still",
            )

        def keyPressEvent(self, event):
            if event.key() == Qt.Key.Key_Escape:
                self.close()

    window = CalibrationTarget()
    window.show()
    app.processEvents()  # render immediately without blocking
    return window, app


if __name__ == "__main__":
    import sys

    uid = sys.argv[1] if len(sys.argv) > 1 else "default"

    target, app = show_calibration_target()
    app.processEvents()

    collect_center_offset(uid, app)
    target.close()
