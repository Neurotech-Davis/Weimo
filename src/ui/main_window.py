"""
ui/main_window.py

MVP main window.
Layout: camera feed (left) | controls + readouts (right)

Right panel sections:
  - Worker Status   : running/stopped indicators per worker
  - Gaze Readout    : live X/Y numeric display
  - Eyetracker Ctrl : camera index, smoothing factor
"""

import cv2
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QGroupBox,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap


class MainWindow(QMainWindow):
    def __init__(self, shared_state):
        super().__init__()
        self.shared_state = shared_state
        self.setWindowTitle("Weimo")
        self.resize(1000, 600)

        # UI-side camera capture (display only)
        self._cap = cv2.VideoCapture(shared_state.camera_index.value)

        self._build_ui()

        # ~30fps poll: update camera feed + readouts
        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        layout.addWidget(self._build_feed_panel(), stretch=3)
        layout.addLayout(self._build_right_panel(), stretch=1)

    def _build_feed_panel(self) -> QLabel:
        self._feed_label = QLabel("Waiting for camera...")
        self._feed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._feed_label.setMinimumSize(640, 480)
        self._feed_label.setStyleSheet("background: #111; color: #555;")
        return self._feed_label

    def _build_right_panel(self) -> QVBoxLayout:
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(10)

        layout.addWidget(self._build_status_group())
        layout.addWidget(self._build_gaze_group())
        layout.addWidget(self._build_eyetracker_controls_group())
        layout.addStretch()

        return layout

    # --- Worker status ---

    def _build_status_group(self) -> QGroupBox:
        box = QGroupBox("Worker Status")
        layout = QVBoxLayout(box)

        self._status_labels = {}
        # Add a row here for each worker you register in main_process.py
        for name in ("eyetracker", "lidar", "classifier"):
            row = QHBoxLayout()
            row.addWidget(QLabel(name))
            indicator = QLabel("●")
            indicator.setStyleSheet("color: #555;")  # grey = unknown
            self._status_labels[name] = indicator
            row.addStretch()
            row.addWidget(indicator)
            layout.addLayout(row)

        return box

    # --- Gaze readout ---

    def _build_gaze_group(self) -> QGroupBox:
        box = QGroupBox("Gaze Position")
        layout = QVBoxLayout(box)

        self._gaze_x_label = QLabel("X:  --")
        self._gaze_y_label = QLabel("Y:  --")
        self._face_label = QLabel("Face: --")

        for lbl in (self._gaze_x_label, self._gaze_y_label, self._face_label):
            lbl.setStyleSheet("font-family: monospace; font-size: 13px;")
            layout.addWidget(lbl)

        return box

    # --- Eyetracker controls ---

    def _build_eyetracker_controls_group(self) -> QGroupBox:
        box = QGroupBox("Eyetracker")
        layout = QVBoxLayout(box)

        # Camera index
        layout.addWidget(QLabel("Camera Index"))
        self._cam_spin = QSpinBox()
        self._cam_spin.setRange(0, 100)
        self._cam_spin.setValue(self.shared_state.camera_index.value)
        self._cam_spin.valueChanged.connect(self._on_camera_changed)
        layout.addWidget(self._cam_spin)

        # Smoothing factor
        layout.addWidget(QLabel("Smoothing"))
        self._smooth_label = QLabel(f"{self.shared_state.smoothing_factor.value:.2f}")
        self._smooth_slider = QSlider(Qt.Orientation.Horizontal)
        self._smooth_slider.setRange(0, 100)
        self._smooth_slider.setValue(
            int(self.shared_state.smoothing_factor.value * 100)
        )
        self._smooth_slider.valueChanged.connect(self._on_smoothing_changed)
        layout.addWidget(self._smooth_slider)
        layout.addWidget(self._smooth_label)

        return box

    # ------------------------------------------------------------------
    # Tick — called every 33ms
    # ------------------------------------------------------------------

    def _tick(self):
        self._update_feed()
        self._update_gaze_readout()
        self._update_worker_status()

    def _update_feed(self):
        ret, frame = self._cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        gaze_x, gaze_y = self.shared_state.get_gaze()

        if gaze_x >= 0 and gaze_y >= 0:
            h, w = frame.shape[:2]
            cx = int((1.0 - gaze_x) * w)
            cy = int(gaze_y * h)
            cv2.circle(frame, (cx, cy), 12, (0, 255, 100), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 100), -1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self._feed_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._feed_label.setPixmap(pixmap)

    def _update_gaze_readout(self):
        gx, gy = self.shared_state.get_gaze()
        face = self.shared_state.face_detected.value

        if gx >= 0:
            self._gaze_x_label.setText(f"X:  {gx:.3f}")
            self._gaze_y_label.setText(f"Y:  {gy:.3f}")
        else:
            self._gaze_x_label.setText("X:  --")
            self._gaze_y_label.setText("Y:  --")

        self._face_label.setText(f"Face: {'detected' if face else 'not detected'}")

    def _update_worker_status(self):
        # Maps worker names to their running flag in shared_state.
        # Add an entry here when you add a new worker + status flag.
        status_map = {
            "eyetracker": self.shared_state.tracker_running.value,
            # "lidar":       self.shared_state.lidar_running.value,
            # "classifier":  self.shared_state.classifier_running.value,
        }
        for name, label in self._status_labels.items():
            running = status_map.get(name, False)
            label.setText("●")
            label.setStyleSheet("color: #00cc66;" if running else "color: #555;")

    # ------------------------------------------------------------------
    # Control callbacks — write params back to shared_state
    # ------------------------------------------------------------------

    def _on_camera_changed(self, value: int):
        self.shared_state.camera_index.value = value
        self._cap.release()
        self._cap = cv2.VideoCapture(value)

    def _on_smoothing_changed(self, value: int):
        factor = value / 100.0
        self.shared_state.smoothing_factor.value = factor
        self._smooth_label.setText(f"{factor:.2f}")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        self._timer.stop()
        self._cap.release()
        self.shared_state.shutdown.set()
        event.accept()
